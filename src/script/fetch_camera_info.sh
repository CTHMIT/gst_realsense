SAVE_DIR="src/config/camera_info"
SAVE_FILE="full_camera_info.txt"

source .env

echo "Ensuring local directory exists: $SAVE_DIR"
mkdir -p "$SAVE_DIR"

echo "Connecting to $ORIN_USER@$ORIN_IP to run 'rs-enumerate-devices -c'..."
OUTPUT_FILE_PATH="$SAVE_DIR/$SAVE_FILE"

sshpass -p "$ORIN_PASS" ssh -o StrictHostKeyChecking=no \
    "$ORIN_USER"@"$ORIN_IP" "rs-enumerate-devices -c" | \
    grep -v ".realsense-config.json" > "$OUTPUT_FILE_PATH"

if [ $? -ne 0 ] || [ ! -s "$OUTPUT_FILE_PATH" ]; then
    echo "SSH connection failed or the file is blank."
    rm -f "$OUTPUT_FILE_PATH"
    exit 1
fi

echo "Automation completed!"
echo "Complete camera information has been saved to: $OUTPUT_FILE_PATH"
ls -l "$OUTPUT_FILE_PATH"