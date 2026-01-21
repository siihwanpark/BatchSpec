#!/usr/bin/env bash
set -u
set -o pipefail

# Import utils.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Register cleanup functions
enable_graceful_exit
register_cleanup 'echo "[CLEANUP] releasing resources..."'

# Set experiment configurations
EXP_FILE="${1:-${SCRIPT_DIR}/exp.tsv}"
[ -f "$EXP_FILE" ] || die "Experiment file not found: $EXP_FILE"

# Define slot configurations
# "GPU_ids|nproc_per_node|rank_group"
declare -A SLOT
SLOT[8g]="0,1,2,3,4,5,6,7|8|0 1 2 3 4 5 6 7"
SLOT[4g0]="0,1,2,3|4|0 1 2 3"
SLOT[4g1]="4,5,6,7|4|0 1 2 3"
SLOT[2g0]="0,1|2|0 1"
SLOT[2g1]="2,3|2|0 1"
SLOT[2g2]="4,5|2|0 1"
SLOT[2g3]="6,7|2|0 1"
SLOT[1g0]="0|1|0"
SLOT[1g1]="1|1|0"
SLOT[1g2]="2|1|0"
SLOT[1g3]="3|1|0"
SLOT[1g4]="4|1|0"
SLOT[1g5]="5|1|0"
SLOT[1g6]="6|1|0"
SLOT[1g7]="7|1|0"

# Create temporary directory
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "${PROJECT_DIR}/tmp" ] || mkdir -p "${PROJECT_DIR}/tmp"
TMP_DIR="$(mktemp -d "${PROJECT_DIR}/tmp/tmp.XXXXXX")"
register_cleanup "rm -rf '$TMP_DIR' || true"

# Normalize experiment file (replace spaces with tabs)
log "Original experiment file: $EXP_FILE"
normalize_tsv "$EXP_FILE" "$TMP_DIR/exp.normalized.tsv"
log "Using normalized experiment file: $TMP_DIR/exp.normalized.tsv"

# Create experiment queue
exp_i=0
while IFS=$'\t' read -r slot model backend draft_length dataset mode bsz prefix_profile num_total_runs; do
    [[ -z "$slot" || "$slot" == "slot" || "$model" == "model" ]] && continue
    exp_i=$((exp_i + 1))
    id=$(printf "t%03d" "$exp_i")
    echo -e "${id}\t${slot}\t${model}\t${backend}\t${draft_length}\t${dataset}\t${mode}\t${bsz}\t${prefix_profile}\t${num_total_runs}" \
        >> "${TMP_DIR}/${slot}.queue"
done < "$TMP_DIR/exp.normalized.tsv"

# Run worker
run_worker() {
    local slot="$1"
    local queue_file="${TMP_DIR}/${slot}.queue"

    [ -f "$queue_file" ] || { warn "No experiments for slot=${slot}"; return 0; }
    [ -n "${SLOT[$slot]:-}" ] || die "Slot not defined: $slot"

    local gpus nproc rank_group
    IFS='|' read -r gpus nproc rank_group <<< "${SLOT[$slot]}"

    log "Worker start slot=$slot gpus=$gpus nproc=$nproc rank_group=$rank_group"

    while IFS=$'\t' read -r id _ model backend draft_length dataset mode bsz prefix_profile num_total_runs; do
		# log "slot=$slot waiting GPUs=$gpus for experiment_id=$id"

		args=( --gpus "$gpus" --nproc "$nproc" --rank_group "$rank_group"
				--model "$model" --backend "$backend" --draft_length "$draft_length"
				--dataset "$dataset" --mode "$mode" --bsz "$bsz" --prefix_profile "$prefix_profile"
				--num_total_runs "$num_total_runs" )

		if [ "$model" != "DSL-8B" ] && [ "$mode" == "sampling" ]; then
			args+=( --top_k 20 )
		fi

		log "slot=$slot running experiment_id=$id: model=$model backend=$backend draft_length=$draft_length dataset=$dataset mode=$mode bsz=$bsz"

		# Logging: write stdout/stderr to tmp first, then move to outdir/output.log
        tmp_log="${TMP_DIR}/${id}.log"
        {
            echo "==== Experiment $id ===="
            echo "slot=$slot gpus=$gpus nproc=$nproc rank_group=$rank_group"
            echo "model=$model backend=$backend draft_length=$draft_length dataset=$dataset mode=$mode"
            echo "bsz=$bsz prefix_profile=$prefix_profile num_total_runs=$num_total_runs"
            echo "started_at=$(date)"
            echo "========================"
            echo

            bash "${SCRIPT_DIR}/main.sh" "${args[@]}"
        } 2>&1 | tee "$tmp_log"
		ret=${PIPESTATUS[0]}

        # Parse [Profiler] Output dir: <path>
        outdir="$(
            sed -n 's/^\[Profiler\] Output dir: //p' "$tmp_log" | tail -n 1 | tr -d $'\r'
        )"

        # If outdir is relative, make it absolute
        if [ -n "$outdir" ] && [ "${outdir:0:1}" != "/" ]; then
            outdir="$(pwd)/$outdir"
        fi

        # Move log file to output directory
        if [ -n "$outdir" ] && [ -d "$outdir" ]; then
            mv -f "$tmp_log" "$outdir/output.log"
            log "slot=$slot finished experiment_id=$id (exit=$ret). log -> $outdir/output.log"
        else
            warn "slot=$slot finished experiment_id=$id (exit=$ret) but outdir not found. log kept at $tmp_log (parsed outdir='$outdir')"
        fi
    done < "$queue_file"

    log "Worker done slot=$slot"
}

# Run partition
run_partition() {
	local part_name="$1"; shift
	local two_g_slot="$1"; shift
	local -a one_g_slots=("$@")

	log "Partition start: $part_name (2g=$two_g_slot, 1g=${one_g_slots[*]})"

	# 1) Run 2 GPU experiments first (if exists)
	if [ -f "${TMP_DIR}/${two_g_slot}.queue" ]; then
		run_worker "$two_g_slot"
	else
		log "Partition $part_name: no queue for 2g slot=$two_g_slot (skip)"
	fi

	# 2) Run 1 GPU experiments in parallel (each GPU is independent) (if exists)
	local -a pids=()
	for slot in "${one_g_slots[@]}"; do
		if [ -f "${TMP_DIR}/${slot}.queue" ]; then
			run_worker "$slot" &
			pids+=("$!")
		else
			log "Partition $part_name: no queue for 1g slot=$slot (skip)"
		fi
	done

	for pid in "${pids[@]}"; do
		wait "$pid"
	done

	log "Partition done: $part_name"
}

# -------------------------
# 8-GPU Scheduling Phases
# -------------------------

# Phase 0: Run 8 GPU experiments alone (highest priority)
if [ -f "${TMP_DIR}/8g.queue" ]; then
    log "Phase 0: running 8g tasks exclusively"
    run_worker 8g
else
    log "Phase 0: no 8g tasks (skip)"
fi

# Run a half partition (either GPUs 0-3 or GPUs 4-7)
run_half() {
    local half_name="$1"; shift
    local four_g_slot="$1"; shift
    local part_a_name="$1"; shift
    local part_a_2g="$1"; shift
    local part_a_1g1="$1"; shift
    local part_a_1g2="$1"; shift
    local part_b_name="$1"; shift
    local part_b_2g="$1"; shift
    local part_b_1g1="$1"; shift
    local part_b_1g2="$1"; shift

    log "Half start: $half_name (4g=$four_g_slot, parts=$part_a_name/$part_b_name)"

    # Step 1) Run 4g slot first (exclusive within this half)
    if [ -f "${TMP_DIR}/${four_g_slot}.queue" ]; then
        log "Half $half_name: running $four_g_slot first"
        run_worker "$four_g_slot"
    else
        log "Half $half_name: no queue for $four_g_slot (skip)"
    fi

    # Step 2) Then run two partitions concurrently within this half
    log "Half $half_name: running $part_a_name and $part_b_name concurrently"
    run_partition "$part_a_name" "$part_a_2g" "$part_a_1g1" "$part_a_1g2" &
    local pid_a=$!

    run_partition "$part_b_name" "$part_b_2g" "$part_b_1g1" "$part_b_1g2" &
    local pid_b=$!

    wait "$pid_a"
    wait "$pid_b"

    log "Half done: $half_name"
}

# Phase 1+2 merged: run left-half and right-half concurrently
#   Left GPUs 0-3: 4g0 -> (pair01, pair23)
#   Right GPUs 4-7: 4g1 -> (pair45, pair67)
log "Phase 1+2: running left-half and right-half concurrently"

run_half "left"  4g0 \
    "pair01" 2g0 1g0 1g1 \
    "pair23" 2g1 1g2 1g3 &
pid_left=$!

run_half "right" 4g1 \
    "pair45" 2g2 1g4 1g5 \
    "pair67" 2g3 1g6 1g7 &
pid_right=$!

wait "$pid_left"
wait "$pid_right"

log "All experiments completed."