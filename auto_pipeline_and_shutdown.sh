#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/endurance/Projects/Elderly Fall Detection"
cd "$PROJECT_DIR"

LOG_FILE="$PROJECT_DIR/auto_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-Pipeline Daemon Started"
echo "Monitoring training process..."
echo "================================================================="

# Wait for run_training.py to finish
while pgrep -f "python3 run_training.py" > /dev/null; do
    sleep 10
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training process has completed!"
sleep 3

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Run deliverables generation (Model exports + Research Figures + Manifest)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing save_all_deliverables.py..."
python3 "$PROJECT_DIR/save_all_deliverables.py"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All deliverables generated and verified!"

# Flush filesystem buffers to disk
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing disks..."
sync
sleep 2

# Shutdown computer cleanly
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutting down computer now as requested..."
systemctl poweroff || busctl call org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager PowerOff "b" true || shutdown -h now
