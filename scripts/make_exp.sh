#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-scripts/exp.tsv}"
mkdir -p "$(dirname "$OUT")"

# TSV header
printf "slot\tmodel\tbackend\tdraft_length\tdataset\tmode\tbsz\tprefix_profile\tnum_total_runs\n" > "$OUT"

append_row() {
  local slot="$1" model="$2" backend="$3" k="$4" dataset="$5" mode="$6" bsz="$7" prefix="$8" runs="$9"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$slot" "$model" "$backend" "$k" "$dataset" "$mode" "$bsz" "$prefix" "$runs" >> "$OUT"
}

for dataset in AIME2025 MMLU-Pro; do
    
    # 4-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        if [ $model == "Qwen3-14B" ]; then
            bsz_list=(256 128 64)
        else
            bsz_list=(256 128)
        fi

        for mode in sampling; do
            for bsz in "${bsz_list[@]}"; do
                if [ $bsz -eq 256 ]; then
                    prefix_profile="short"
                else
                    prefix_profile="long"
                fi

                append_row "4g" "$model" "mtp" 1 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "4g" "$model" "mtp" 2 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "4g" "$model" "mtp" 3 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "4g" "$model" "mtp" 4 "$dataset" "$mode" "$bsz" "$prefix_profile" 6

            done
        done
    done

    # 2-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        if [ $model == "Qwen3-14B" ]; then
            bsz_list=(32)
        else
            bsz_list=(64)
        fi

        for mode in sampling; do
            for bsz in "${bsz_list[@]}"; do
                prefix_profile="long"

                append_row "2g0" "$model" "mtp" 1 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "2g1" "$model" "mtp" 2 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "2g1" "$model" "mtp" 3 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "2g0" "$model" "mtp" 4 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
        
            done
        done
    done

    # 1-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        if [ $model == "Qwen3-14B" ]; then
            bsz_list=(16)
        else
            bsz_list=(32 16)
        fi

        for mode in sampling; do
            for bsz in "${bsz_list[@]}"; do
                prefix_profile="long"

                append_row "1g0" "$model" "mtp" 1 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "1g1" "$model" "mtp" 2 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "1g2" "$model" "mtp" 3 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
                append_row "1g3" "$model" "mtp" 4 "$dataset" "$mode" "$bsz" "$prefix_profile" 6
            done
        done
    done

done

echo "Wrote: $OUT"