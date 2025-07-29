#!/bin/bash
set -euo pipefail

# Set log directory
LOG_DIR="./inference_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Inference tasks started: $START_TIME" | tee -a "$LOG_DIR/inference_all.log"

MODELS=("gpt-4o-mini-2024-07-18" "claude-3-5-sonnet-20240620" “claude-3-7-sonnet-20250219” "gemini-2.0-flash" "gemini-2.5-flash" "deepseek-chat" "deepseek-reasoner" "Llama-3.3-70B-Instruct" "Llama-3.1-70B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct")

for MODEL in "${MODELS[@]}"; do
    echo "=======================================" | tee -a "$LOG_DIR/inference_all.log"
    echo "Running inference: model=$MODEL" | tee -a "$LOG_DIR/inference_all.log"

    python ./script/inference.py --model "$MODEL" --log-dir "$LOG_DIR" --run-all 2>&1 | tee -a "$LOG_DIR/inference_$MODEL.log"
    
    if [ $? -eq 0 ]; then
        echo "Inference succeeded: model=$MODEL" | tee -a "$LOG_DIR/inference_all.log"
    else
        echo "❌ Inference failed: model=$MODEL" | tee -a "$LOG_DIR/inference_all.log"
        echo "🛑 Aborting inference due to failure." | tee -a "$LOG_DIR/inference_all.log"
        exit 1
    fi
    echo "=======================================" | tee -a "$LOG_DIR/inference_all.log"
    echo "" | tee -a "$LOG_DIR/inference_all.log"
done

# Record end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "All inference tasks completed: $END_TIME" | tee -a "$LOG_DIR/inference_all.log"

START_SECONDS=$(date -d "$START_TIME" +%s)
END_SECONDS=$(date -d "$END_TIME" +%s)
DURATION=$((END_SECONDS - START_SECONDS))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_DIR/inference_all.log" 