#!/bin/bash
# Capture live detection data and create scientific visualization

OUTPUT_DIR="results/unsupervised/real_can0"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "CAN IDS Real-Time Detection Capture"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Run the detector in debug mode"
echo "  2. Capture output to file"
echo "  3. Create scientific visualization"
echo ""
echo "Press Ctrl+C when done capturing..."
echo ""

# Capture debug output
python scripts/deploy_realtime_debug.py 2>&1 | tee "${OUTPUT_DIR}/capture_$(date +%Y%m%d_%H%M%S).txt"

# Create visualization
if [ -f "${OUTPUT_DIR}/capture_*.txt" ]; then
    LATEST_CAPTURE=$(ls -t ${OUTPUT_DIR}/capture_*.txt | head -1)
    echo ""
    echo "Creating visualization from: $LATEST_CAPTURE"
    python scripts/visualize_normal_vs_attack.py \
        --input "$LATEST_CAPTURE" \
        --output "${OUTPUT_DIR}/normal_vs_attack_detection.png"
    echo ""
    echo "✓ Visualization saved to: ${OUTPUT_DIR}/normal_vs_attack_detection.png"
else
    echo "No capture file found"
fi

