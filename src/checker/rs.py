import pyrealsense2 as rs
try:
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"發現 {len(devices)} 個裝置")
    for dev in devices:
        print(f"  - 名稱: {dev.get_info(rs.camera_info.name)}")
        print(f"  - 序號: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  - USB類型: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
except Exception as e:
    print(f"錯誤: {e}")