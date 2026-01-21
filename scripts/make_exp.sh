#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-scripts/exp.tsv}"
mkdir -p "$(dirname "$OUT")"

# TSV header
printf "slot\tmodel\tbackend\tdraft_length\tdataset\tmode\tbsz\tprefix_profile\tnum_total_runs\n" > "$OUT"

append_row() {
  local slot="$1" model="$2" backend="$3" k="$4" dataset="$5" mode="$6" bsz="$7" prefix="$8" runs="$9"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$slot" "$model" "$backend" "$k" "$dataset" "sampling" "$bsz" "$prefix" "$runs" >> "$OUT"
}

num_total_runs=4
for dataset in AIME2025 MMLU-Pro CodeForces; do
    
    # 8-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        bsz_list=(256 128 64)
        for bsz in "${bsz_list[@]}"; do
            if [ $bsz -eq 256 ]; then
                if [ $model == "Qwen3-14B" ]; then
                    prefix_profile="12k"
                else
                    prefix_profile="16k"
                fi
            elif [ $bsz -eq 128 ]; then
                prefix_profile="24k"
            elif [ $bsz -eq 64 ]; then
                prefix_profile="32k"
            fi

            # standard
            if [ $dataset == "AIME2025" ]; then
                append_row "8g" "$model" "standard" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            fi

            # ngram
            append_row "8g" "$model" "ngram" 5 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "8g" "$model" "ngram" 10 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"

            for backend in eagle magicdec mtp; do
                append_row "8g" "$model" "$backend" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
                append_row "8g" "$model" "$backend" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
                append_row "8g" "$model" "$backend" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
                append_row "8g" "$model" "$backend" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            done
        done
    done

    # 8-GPU Standalone 
    for model in Qwen3-8B Qwen3-14B; do
        bsz_list=(256 128 64)
        for bsz in "${bsz_list[@]}"; do
            if [ $bsz -eq 256 ]; then
                prefix_profile="8k"
            elif [ $bsz -eq 128 ]; then
                prefix_profile="16k"
            elif [ $bsz -eq 64 ]; then
                if [ $model == "Qwen3-14B" ]; then
                    prefix_profile="28k"
                else
                    prefix_profile="32k"
                fi
            fi
            
            append_row "8g" "$model" "standalone" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "8g" "$model" "standalone" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "8g" "$model" "standalone" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "8g" "$model" "standalone" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"

        done
    done

    # 4-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        bsz=32
        prefix_profile="32k"

        # standard
        if [ $dataset == "AIME2025" ]; then
            append_row "4g0" "$model" "standard" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        fi

        # ngram
        append_row "4g0" "$model" "ngram" 5 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "4g1" "$model" "ngram" 10 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        
        for backend in standalone eagle magicdec mtp;do
            append_row "4g0" "$model" "$backend" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "4g1" "$model" "$backend" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "4g1" "$model" "$backend" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "4g0" "$model" "$backend" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        done
    done

    # 4-GPU standalone runs
    for model in Qwen3-8B Qwen3-14B; do
        bsz=32
        prefix_profile="32k"

        append_row "4g0" "$model" "standalone" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "4g1" "$model" "standalone" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "4g1" "$model" "standalone" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "4g0" "$model" "standalone" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
    done


    # 2-GPU runs
    for model in Qwen3-8B DSL-8B Qwen3-14B; do
        bsz=16
        prefix_profile="32k"

        # standard
        if [ $dataset == "AIME2025" ]; then
            append_row "2g0" "$model" "standard" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        fi

        # ngram
        append_row "2g1" "$model" "ngram" 5 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "2g2" "$model" "ngram" 10 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        
        for backend in eagle magicdec mtp;do
            append_row "2g0" "$model" "$backend" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "2g1" "$model" "$backend" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "2g2" "$model" "$backend" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
            append_row "2g3" "$model" "$backend" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        done
    done

    # 2-GPU standalone runs
    for model in Qwen3-8B Qwen3-14B; do
        bsz=16
        if [ $model == "Qwen3-8B" ]; then
            prefix_profile="32k"
        else
            prefix_profile="28k"
        fi

        append_row "2g0" "$model" "standalone" 1 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "2g1" "$model" "standalone" 2 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "2g2" "$model" "standalone" 3 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
        append_row "2g3" "$model" "standalone" 4 "$dataset" "sampling" "$bsz" "$prefix_profile" "$num_total_runs"
    done
done

echo "Wrote: $OUT"