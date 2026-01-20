#!/bin/bash

USAGE="""
    python -m tools.aggregate_profiler_reports \
        --input_root <profiler_out_dir>\
        --output_root <profiler_summary_dir>\
        --reference_dataset_name <reference_dataset_name> (e.g. AIME2025)\
        --summary_mode <full|compact>\
        --k_select <throughput|lat_mean|lat_p95>\
        [--details_include_policies]
"""

INPUT_ROOT="profiler_out"
OUTPUT_ROOT="profiler_summary"
REFERENCE_DATASET_NAME="AIME2025"
SUMMARY_MODE="full"
K_SELECT="throughput"
DETAILS_INCLUDE_POLICIES="--details_include_policies"

python -m tools.aggregate_profiler_reports \
    --input_root $INPUT_ROOT \
    --output_root $OUTPUT_ROOT \
    --reference_dataset_name $REFERENCE_DATASET_NAME \
    --summary_mode $SUMMARY_MODE \
    --k_select $K_SELECT \
    $DETAILS_INCLUDE_POLICIES