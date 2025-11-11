v4l2-ctl --list-devices

for d in /dev/video*; do
  v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null # | grep -q "Z16" && echo "$d"
done

for d in /dev/video*; do
  if v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null # | grep -q "Z16"; then
    echo "== $d ==" 
    v4l2-ctl -d "$d" --list-formats-ext # | sed -n '/Z16/,+15p'
  fi
done
