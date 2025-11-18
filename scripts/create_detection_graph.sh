#!/bin/bash
# Helper script to capture debug output and create visualization

OUTPUT_DIR="results/unsupervised/real_can0"
mkdir -p "$OUTPUT_DIR"

echo "Run the debug script and save output to a file:"
echo "  python scripts/deploy_realtime_debug.py > ${OUTPUT_DIR}/debug_output.txt 2>&1"
echo ""
echo "Then create visualization:"
echo "  python scripts/visualize_detection.py --input ${OUTPUT_DIR}/debug_output.txt --output ${OUTPUT_DIR}/detection_performance.png"

