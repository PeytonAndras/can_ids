#!/bin/bash
# Test script for rate-based detection

set -e

echo "=========================================="
echo "Rate-Based Detection Test"
echo "=========================================="
echo ""

# Check if can0 interface exists
if ! ip link show can0 &>/dev/null; then
    echo "⚠️  Warning: can0 interface not found"
    echo "   Make sure your CAN interface is set up"
    echo ""
fi

# Check if config has rate detection enabled
CONFIG_FILE="deployment/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found"
    exit 1
fi

if grep -q "enabled: true" "$CONFIG_FILE" | grep -A 1 "rate_detection"; then
    echo "✓ Rate detection is enabled in config"
else
    echo "⚠️  Warning: Rate detection may not be enabled"
    echo "   Check deployment/config.yaml"
fi

echo ""
echo "Test Options:"
echo "1. Test with your cycling attack script"
echo "2. Test with a simple rate anomaly (high rate)"
echo "3. Test with regular timing pattern"
echo ""
read -p "Choose test (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Testing with cycling attack script..."
        echo "Make sure your attack script is ready to run"
        echo ""
        echo "In another terminal, run:"
        echo "  ./your_attack_script.sh"
        echo ""
        echo "Then press Enter here to start the IDS..."
        read
        python3 scripts/deploy_realtime.py --can-channel can0
        ;;
    2)
        echo ""
        echo "Testing high-rate attack..."
        echo "This will send messages at high rate for 5 seconds"
        echo ""
        echo "Starting IDS in background..."
        python3 scripts/deploy_realtime.py --can-channel can0 > /tmp/ids_output.log 2>&1 &
        IDS_PID=$!
        sleep 2
        
        echo "Injecting high-rate messages..."
        for i in {1..500}; do
            cansend can0 039#00003A0DD87D5C7A
            sleep 0.001  # 1000 msg/s
        done
        
        sleep 2
        kill $IDS_PID 2>/dev/null || true
        echo ""
        echo "Check /tmp/ids_output.log for alerts"
        ;;
    3)
        echo ""
        echo "Testing regular timing pattern..."
        echo "This will send messages at exactly 0.1s intervals"
        echo ""
        echo "Starting IDS in background..."
        python3 scripts/deploy_realtime.py --can-channel can0 > /tmp/ids_output.log 2>&1 &
        IDS_PID=$!
        sleep 2
        
        echo "Injecting regular pattern (0.1s intervals)..."
        for i in {1..50}; do
            cansend can0 039#00003A0DD87D5C7A
            sleep 0.1
        done
        
        sleep 2
        kill $IDS_PID 2>/dev/null || true
        echo ""
        echo "Check /tmp/ids_output.log for alerts"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

