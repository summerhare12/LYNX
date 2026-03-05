#!/bin/bash

# 定义日志文件
LOG_FILE="run_all_tasks.log"
MODEL_NAME="deepseek-chat"
# 转换模型名以匹配 Python 脚本中的文件夹命名规则 (output_deepseek-chat 或 output_deepseek_chat)
# 假设你的代码里是 model.replace(":","_")，如果是 deepseek-chat 则目录通常是 output_deepseek-chat
OUTPUT_DIR_NAME="output_${MODEL_NAME//:/_}" 

echo "=== Starting Full Benchmark Run at $(date) ===" | tee -a "$LOG_FILE"

# 函数：运行脚本并清理缓存
run_script() {
    script_name=$1
    echo "---------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running $script_name..." | tee -a "$LOG_FILE"
    
    # 运行脚本
    python3 "$script_name" --model "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $script_name finished successfully." | tee -a "$LOG_FILE"
    else
        echo "FAILURE: $script_name failed with error code $?." | tee -a "$LOG_FILE"
    fi

    # === 新增逻辑：删除 task_progress.txt ===
    # 假设 task_progress.txt 位于 ./Spider2/spider2-lite/output_deepseek-chat/ 目录下
    # Python 脚本中的 BASE_DIR 是 "./Spider2/spider2-lite"
    PROGRESS_FILE="./Spider2/spider2-lite/${OUTPUT_DIR_NAME}/task_progress.txt"
    
    if [ -f "$PROGRESS_FILE" ]; then
        echo "Removing progress file: $PROGRESS_FILE" | tee -a "$LOG_FILE"
        rm "$PROGRESS_FILE"
    else
        echo "Progress file not found: $PROGRESS_FILE (maybe already deleted or never created)" | tee -a "$LOG_FILE"
    fi
}

# 1. 运行 BigQuery 任务
run_script "bigquery.py"

# 2. 运行 Snowflake 任务
run_script "run_snowflake.py"

# 3. 运行 Local (SQLite) 任务
run_script "local.py"

echo "---------------------------------------------------" | tee -a "$LOG_FILE"
echo "=== Benchmark Run Loop Completed at $(date) ===" | tee -a "$LOG_FILE"