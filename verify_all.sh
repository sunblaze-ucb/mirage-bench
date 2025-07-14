#!/bin/bash
set -euo pipefail

# Set log directory
LOG_DIR="verify_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p $LOG_DIR

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Verification tasks started: $START_TIME" | tee -a "$LOG_DIR/verify_all.log"

# Define verification types and corresponding scenarios
declare -A TYPE_SCENARIOS
TYPE_SCENARIOS["unexpected_transition"]="theagentcompany webarena"
TYPE_SCENARIOS["users_questions"]="theagentcompany taubench"
TYPE_SCENARIOS["misleading"]="swebench webarena"
TYPE_SCENARIOS["repetitive_4"]="webarena workarena swebench"
TYPE_SCENARIOS["repetitive_7"]="webarena workarena swebench"
TYPE_SCENARIOS["popup"]="webarena osworld"
TYPE_SCENARIOS["underspecified"]="webarena osworld"
TYPE_SCENARIOS["unachievable"]="webarena workarena osworld"
TYPE_SCENARIOS["error_feedback"]="webarena workarena swebench"

MODELS=("gpt-4o-mini-2024-07-18" "claude-3-5-sonnet-20240620" "gemini-2.0-flash" "gemini-2.5-flash" "deepseek-chat" "deepseek-reasoner" "Llama-3.3-70B-Instruct" "Llama-3.1-70B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct")

# Execute verification tasks
for TYPE in "${!TYPE_SCENARIOS[@]}"; do
  for SCENARIO in ${TYPE_SCENARIOS[$TYPE]}; do
    for MODEL in "${MODELS[@]}"; do
      echo "=======================================" | tee -a "$LOG_DIR/verify_all.log"
      echo "Running verification: type=$TYPE, scenario=$SCENARIO, model=$MODEL" | tee -a "$LOG_DIR/verify_all.log"
      
      # Run verification command
      python ./script/verifier.py --type "$TYPE" --scenario "$SCENARIO" --model "$MODEL" 2>&1 | tee -a "$LOG_DIR/verify_$TYPE_$SCENARIO_$MODEL.log"
      
      # Check execution status
      if [ $? -eq 0 ]; then
        echo "Verification succeeded: type=$TYPE, scenario=$SCENARIO, model=$MODEL" | tee -a "$LOG_DIR/verify_all.log"
      else
        echo "❌ Verification failed: type=$TYPE, scenario=$SCENARIO, model=$MODEL" | tee -a "$LOG_DIR/verify_all.log"
        echo "🛑 Aborting verification due to failure." | tee -a "$LOG_DIR/verify_all.log"
        exit 1
      fi
      echo "=======================================" | tee -a "$LOG_DIR/verify_all.log"
      echo "" | tee -a "$LOG_DIR/verify_all.log"
    done
  done
done

# Record end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "All verification tasks completed: $END_TIME" | tee -a "$LOG_DIR/verify_all.log"

# Calculate total runtime
START_SECONDS=$(date -d "$START_TIME" +%s)
END_SECONDS=$(date -d "$END_TIME" +%s)
DURATION=$((END_SECONDS - START_SECONDS))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_DIR/verify_all.log" 