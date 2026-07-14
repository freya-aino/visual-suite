import time
import hid

# List all connected HID devices
for device in hid.enumerate():
    print(f"VID: {device['vendor_id']:#06x}  PID: {device['product_id']:#06x}  "
          f"Usage: {device['usage']:#06x}  Name: {device['product_string']}")


VENDOR_ID  = 0x0079  # Replace with your controller's VID
PRODUCT_ID = 0x0006  # Replace with your controller's PID

# Open the device
gamepad = hid.device()
gamepad.open(VENDOR_ID, PRODUCT_ID)
gamepad.set_nonblocking(True)

print(f"Connected to: {gamepad.get_manufacturer_string()} - {gamepad.get_product_string()}")

try:
    # --- LED ON ---
    # Most generic controllers accept a report like [report_id, led_byte]
    # 0x00 = report ID (try 0x00 or 0x01 depending on your device)
    # 0xFF = all LEDs on (adjust bitmask per your controller's protocol)
    report_on  = [0x00, 0xFF] + [0x00] * 62  # 64 bytes total
    report_off = [0x00, 0x00] + [0x00] * 62

    print("LEDs ON")
    gamepad.write(report_on)
    time.sleep(2)

    print("LEDs OFF")
    gamepad.write(report_off)
    time.sleep(1)

finally:
    gamepad.close()