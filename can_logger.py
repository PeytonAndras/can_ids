import can
import csv
import time

# This logger writes CAN frames to CSV and auto-labels periods as 'attack' or 'normal'.
# Marker semantics:
# - Any frame with arbitration ID 0x010 is treated as a marker.
# - If the first data byte is 0x01 -> attack begins (label=attack).
# - If the first data byte is 0x00 -> attack ends (label=normal).
# - Otherwise (no payload or other values) -> toggle current state.
# The marker frame itself is logged with the label resulting AFTER applying the new state.

# Configure CAN bus (Linux SocketCAN)
bus = can.interface.Bus(channel='can0', interface='socketcan')

MARKER_CAN_ID = 0x010
running = True
attack_active = False

try:
    with open('can_log_3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'can_id', 'dlc', 'data_hex', 'scenario', 'is_attack'])

        while running:
            msg = bus.recv()  # blocking wait
            if msg is None:
                continue

            # Apply marker logic if this is the designated marker ID
            if msg.arbitration_id == MARKER_CAN_ID:
                # Determine new state from payload if provided, else toggle
                if msg.dlc and len(msg.data) > 0:
                    first = msg.data[0]
                    if first == 0x01:
                        attack_active = True
                    elif first == 0x00:
                        attack_active = False
                    else:
                        attack_active = not attack_active
                else:
                    attack_active = not attack_active

            scenario = 'attack' if attack_active else 'normal'
            is_attack = 1 if attack_active else 0

            writer.writerow([
                msg.timestamp,
                hex(msg.arbitration_id),
                msg.dlc,
                msg.data.hex(),
                scenario,
                is_attack,
            ])

            # Flush promptly so markers/labels are not lost on exit
            f.flush()

except KeyboardInterrupt:
    running = False
finally:
    try:
        bus.shutdown()
    except Exception:
        pass

# Example output:
# timestamp,can_id,dlc,data_hex,scenario,is_attack
# 1700000000,0x6d,8,00000aaa607da8ed,normal,0
# 1700000010,0x10,1,01,attack,1 <- marker: start
# 1700000011,0x7d,8,00000bbb70b8c9fe,attack,1
# 1700000020,0x10,1,00,normal,0 <- marker: stop
# 1700000021,0x8e,8,00000ccc80c9dafe,normal,0