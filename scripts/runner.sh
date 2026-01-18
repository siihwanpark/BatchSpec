#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

enable_graceful_exit
register_cleanup 'echo "[CLEANUP] releasing resources..."'
register_cleanup 'pkill -P $$ || true'

EXP_FILE="${1:-${SCRIPT_DIR}/exp.tsv}"
[ -f "$EXP_FILE" ] || die "Experiment file not found: $EXP_FILE"

log "Original experiment file: $EXP_FILE"
NORMALIZED_EXP_FILE="$(normalize_tsv "$EXP_FILE")"
log "Using normalized experiment file: $NORMALIZED_EXP_FILE"

declare -A SLOT
SLOT[4g]="0,1,2,3 4 '0 1 2 3'"
SLOT[2g0]="0,1 2 '0 1'"
SLOT[2g1]="2,3 2 '0 1'"
SLOT[1g0]="0 1 '0'"
SLOT[1g1]="1 1 '0'"
SLOT[1g2]="2 1 '0'"
SLOT[1g3]="3 1 '0'"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "${PROJECT_DIR}/tmp" ] || mkdir -p "${PROJECT_DIR}/tmp"
TMP_DIR="$(mktemp -d "${PROJECT_DIR}/tmp/tmp.XXXXXX")"
register_cleanup "rm -rf '$TMP_DIR' || true"

exp_i=0
while IFS=$'\t' read -r slot model backend draft_length dataset mode bsz prefix_profile num_total_runs; do
    [[ -z "$slot" || "$slot" == "slot" || "$model" == "model" ]] && continue
    exp_i=$((exp_i + 1))
    id=$(printf "t%03d" "$exp_i")
    echo -e "${id}\t${slot}\t${model}\t${backend}\t${draft_length}\t${dataset}\t${mode}\t${bsz}\t${prefix_profile}\t${num_total_runs}" \
        >> "${TMP_DIR}/${slot}.queue"
done < "$NORMALIZED_EXP_FILE"

run_worker() {
    local slot="$1"
    local queue_file="${TMP_DIR}/${slot}.queue"

    [ -f "$queue_file" ] || { warn "No experiments for slot=${slot}"; return 0; }
    [ -n "${SLOT[$slot]:-}" ] || die "Slot not defined: $slot"

    local gpus nproc rank_group
    eval "set -- ${SLOT[$slot]}"
    gpus="$1"; nproc="$2"; rank_group="$3"

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
		# bash "${SCRIPT_DIR}/main.sh" "${args[@]}"
		# log "slot=$slot finished experiment_id=$id"

		# ------------------------------------------------------------
        # Logging: write stdout/stderr to tmp first, then move to outdir/output.log
        # ------------------------------------------------------------
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

        # Parse: [Profiler] Output dir: <path>
        outdir="$(
            sed -n 's/^\[Profiler\] Output dir: //p' "$tmp_log" | tail -n 1 | tr -d $'\r'
        )"

        # If outdir is relative, make it absolute (safer)
        if [ -n "$outdir" ] && [ "${outdir:0:1}" != "/" ]; then
            outdir="$(pwd)/$outdir"
        fi

        if [ -n "$outdir" ] && [ -d "$outdir" ]; then
            mv -f "$tmp_log" "$outdir/output.log"
            log "slot=$slot finished experiment_id=$id (exit=$ret). log -> $outdir/output.log"
        else
            warn "slot=$slot finished experiment_id=$id (exit=$ret) but outdir not found. log kept at $tmp_log (parsed outdir='$outdir')"
        fi
    done < "$queue_file"

    log "Worker done slot=$slot"
}

run_partition() {
	local part_name="$1"; shift
	local two_g_slot="$1"; shift
	local -a one_g_slots=("$@")

	log "Partition start: $part_name (2g=$two_g_slot, 1g=${one_g_slots[*]})"

	# 1) 2 GPU experiments first (if exists)
	if [ -f "${TMP_DIR}/${two_g_slot}.queue" ]; then
		run_worker "$two_g_slot"
	else
		log "Partition $part_name: no queue for 2g slot=$two_g_slot (skip)"
	fi

	# 2) 1 GPU experiments in parallel (each GPU is independent) (if exists)
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

# ------------------------------------------------------------
# Phase 0: 4g runs alone (highest priority)
# ------------------------------------------------------------
if [ -f "${TMP_DIR}/4g.queue" ]; then
	log "Phase 0: running 4g tasks exclusively"
	run_worker 4g
else
	log "Phase 0: no 4g tasks (skip)"
fi

# ------------------------------------------------------------
# Phase 1: two independent partitions run concurrently
#   - pair01: 2g0 then (1g0,1g1) in parallel
#   - pair23: 2g1 then (1g2,1g3) in parallel
# ------------------------------------------------------------
log "Phase 1: running pair01 and pair23 concurrently"

run_partition "pair01" 2g0 1g0 1g1 &
pid01=$!

run_partition "pair23" 2g1 1g2 1g3 &
pid23=$!

wait "$pid01"
wait "$pid23"

log "All experiments completed."
