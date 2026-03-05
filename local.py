import os
import re
import pandas as pd
import json
import sqlite3 
import argparse
from localagent import AutoCodingAgent
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

BASE_DIR = "./Spider2/spider2-lite"
DATA_DIR = os.path.join(BASE_DIR, "resource/databases/sqlite")
TASK_FILE = "./Spider2/spider2-lite/spider2-lite.jsonl"


MAX_TRIES = 25

def get_metadata_path(instance_id, db_id):
    base_db_path = os.path.join(DATA_DIR, db_id)
    if not os.path.exists(base_db_path):
        print(f"[Warn] DB path not found: {base_db_path}")
        return []

    found_paths = []

    for root, dirs, files in os.walk(base_db_path):
        if "DDL.csv" in files:
            found_paths.append(root)
            
    if not found_paths:
        print(f"[Warn] No metadata directory found in {base_db_path}")
    return found_paths

# def get_metadata_path(instance_id, db_id):
#     base_db_path = os.path.join(DATA_DIR, db_id)   
#     if not os.path.exists(base_db_path):
#         print(f"[Warn] DB path not found: {base_db_path}")
#         return None   

#     subdirs = [d for d in os.listdir(base_db_path) if os.path.isdir(os.path.join(base_db_path, d))]  
#     if not subdirs:
#         print(f"[Warn] No subdirectory found in {base_db_path}")
#         return None
    
#     target_dir = os.path.join(base_db_path, subdirs[0])
#     return target_dir

def external_knowledge_path(instance_id, knowledge_filename):
    if not knowledge_filename:
        return None
        
    path = os.path.join(BASE_DIR, "resource/documents",knowledge_filename)
    if os.path.exists(path):
        return path
    else:
        return None

def execute_sql(db_path, sql_query):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql_query, conn)

def run_task(agent, task, output_dir):
    instance_id = task['instance_id']
    instruction = task['question']
    db_id = task['db']
    knowledge_file = task['external_knowledge']
    sql_query = ""
    error = ""
    exe_flag = False
    db_path = os.path.join(BASE_DIR, "resource/databases/local", f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"[Error] Database file not found: {db_path}")
        return "Database Not Found", 0
    
    print(f"\n=== Running Task: {instance_id} ===")
    print(f"Query: {instruction}")
    
    metadata_dir = get_metadata_path(instance_id, db_id)
    knowledge_path = external_knowledge_path(instance_id, knowledge_file)
    # print(f"metadata_dir: {metadata_dir}, external_knowledge_path: {knowledge_path}\n")
    if not metadata_dir:
        return "Metadata Not Found", 0

    output_path = os.path.join(output_dir, f"{instance_id}.csv")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for attempt in range(1, MAX_TRIES + 1):
        print(f"\n--- attempt {attempt} ---\n")
        sql_query, token_use = agent.run(instruction, knowledge_path, metadata_dir, attempt, error, exe_flag)
        token_record = os.path.join(output_dir, f"tokens.txt")
        with open(token_record, 'a', encoding='utf-8') as f:
            f.write(f"Task {instance_id}: {token_use}")
        
        if not sql_query:
            error = "Query is empty. Please write a single valid SQlite SQL query to retrieve the right answer."
            print(error)
            continue
        try:
            df = func_timeout(300, execute_sql, args=(db_path, sql_query))     
            
            if re.match(r"^\s*SELECT\s+\*", sql_query, re.IGNORECASE):
                print("Detected exploratory query (SELECT *). Treating as intermediate step.")
                error = (
                    f"The query executed successfully and returned {len(df)} rows. "
                    f"Here is a sample of the data:\n{df.head(3).to_string()}\n\n"
                    "This seems to be an exploratory query (SELECT *). "
                    "Please write the FINAL SQL query to answer the user's question strictly."
                )
                continue

            if df.empty:
                error = "Query executed successfully but returned no data (Empty DataFrame).\n"
                print(error)
            # elif exe_flag is False:
            #     error = f"The query executed successfully and returned {len(df)} rows. Here is a sample of the returned data:\n{df.head(5).to_string()}\n"
            #     exe_flag = True
            #     continue
            else:
                df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")

                output_sql = os.path.join(output_dir, f"{instance_id}.txt")
                with open(output_sql, 'w', encoding='utf-8') as f:
                    f.write(sql_query)
                
                return "Success!", attempt
            
        except FunctionTimedOut:
            print(f"[Error] SQL execution timed out!")
            error = "SQL execution timed out!"
            exe_flag = False
        
        except Exception as e:
            print(f"SQL Error: {e}")
            error = str(e)
            exe_flag = False
    
    pd.DataFrame().to_csv(output_path, index=False)
    print(f"Failed after{MAX_TRIES} attempts. Empty CSV saved to {output_path}")

    output_sql = os.path.join(output_dir, f"{instance_id}.txt")
    with open(output_sql, 'w', encoding='utf-8') as f:
        f.write(sql_query)

    return "Fail.", MAX_TRIES



def main(): 
    parser = argparse.ArgumentParser(description="Run Spider2-Lite tasks with a specific model.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name to use (e.g., deepseek-chat, qwen3:14b)")
    args = parser.parse_args()  
    model_name = args.model.replace(":","_")
    output_subdir = f"output_{model_name}"
    output_dir = os.path.join(BASE_DIR, output_subdir)
    cache_file = os.path.join(BASE_DIR, output_subdir, "task_progress.txt")
    log_file = os.path.join(BASE_DIR, output_subdir, "task_log.csv")

    print(f"Initializing AutoCodingAgent with model: {args.model}")
    agent = AutoCodingAgent(model_name=args.model) 
    
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("instance_id,status,attempts\n")

    tasks = []
    with open(TASK_FILE, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    last_success_id = None
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            last_success_id = f.read().strip()
            print(f"Resuming from after task: {last_success_id}")

    skipping = True if last_success_id else False

    for task in tqdm(tasks):
        if skipping:
            if task['instance_id'] == last_success_id:
                skipping = False
            continue
        if "local" in task['instance_id']:
            status, attempts = run_task(agent, task, output_dir)
            print(f"Task {task['instance_id']} finished with status: {status}, attempts: {attempts}")

            with open(log_file, 'a') as f:
                f.write(f"{task['instance_id']},{status},{attempts}\n")

            with open(cache_file, 'w') as f:
                f.write(task['instance_id'])

    
if __name__ == "__main__":
    main()



