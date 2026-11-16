#!/bin/bash
# start_securecheck.sh
# ----------------------
# Double-click this file (choose "Execute in Terminal" if asked) to
# start the whole SecureCheck system - dashboard + camera - with one
# click. This is a plain shell script rather than a .desktop file,
# so there's no special launcher format for the file manager to
# misinterpret.

# Move into this script's own folder first, so it works no matter
# where it's double-clicked from.
cd "$(dirname "$0")"

echo "Starting SecureCheck from: $(pwd)"
python3 launch_system.py

echo ""
echo "Press Enter to close this window..."
read
