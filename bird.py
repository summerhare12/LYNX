import os
import re
import pandas as pd
import json
import sqlite3 
import argparse
# from birdagent import AutoCodingAgent
from birdds import AutoCodingAgent
from birdtools import BirdMetaCache
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

BASE_DIR = "./BIRD"
DB_ROOT_DIR = os.path.join(BASE_DIR, "dev_databases")
TASK_FILE = os.path.join(BASE_DIR, "mini_dev_sqlite.json") 


MAX_TRIES = 5

def execute_sql(db_path, sql_query):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql_query, conn)

def run_task(agent, task, meta_cache, output_dir):
    question_id = str(task['question_id'])
    instruction = task['question']
    db_id = task['db_id']
    evidence = task['evidence']

    sql_query = ""
    error = ""
    exe_flag = False
    empty_flag = False
    db_path = os.path.join(DB_ROOT_DIR, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"[Error] Database file not found: {db_path}")
        return "Database Not Found", 0
    
    print(f"\n=== Running Task: {question_id} (DB: {db_id}) ===")
    print(f"Query: {instruction}")

    metadata_str, table_names, column_json = meta_cache.get_database_meta(db_id)
    # schema_info = f"Table names: {table_names}\n Column JSON: {column_json}\n"
    schema_info = ""

    output_path = os.path.join(output_dir, f"{question_id}.csv")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for attempt in range(1, MAX_TRIES + 1):
        print(f"\n--- attempt {attempt} ---\n")
        sql_query, token_use = agent.run(instruction, evidence, metadata_str, schema_info, attempt, error, exe_flag)
        token_record = os.path.join(output_dir, f"tokens.txt")
        with open(token_record, 'a', encoding='utf-8') as f:
            f.write(f"Task {question_id}: {token_use}")

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

            if df.empty and empty_flag is False:
                error = "Query executed successfully but returned an empty DataFrame.\n"
                print(error)
                empty_flag = True
            elif df.empty and empty_flag:
                pd.DataFrame().to_csv(output_path, index=False)
                print(f"Empty result confirmed. Results saved to {output_path}")

                output_sql = os.path.join(output_dir, f"{question_id}.sql")
                with open(output_sql, 'w', encoding='utf-8') as f:
                    f.write(sql_query)
                return "Success! (Empty Result)", attempt

            elif exe_flag is False:
                error = f"The query executed successfully and returned {len(df)} rows. Here is a sample of the data:\n{df.head(5).to_string()}\n"
                exe_flag = True
                continue
            else:
                df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")

                output_sql = os.path.join(output_dir, f"{question_id}.sql")
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

    output_sql = os.path.join(output_dir, f"{question_id}.sql")
    with open(output_sql, 'w', encoding='utf-8') as f:
        f.write(sql_query)

    return "Fail.", MAX_TRIES



def main(): 
    parser = argparse.ArgumentParser(description="Run Spider2-Lite tasks with a specific model.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name to use (e.g., deepseek-chat, qwen3:14b)")
    args = parser.parse_args()  
    model_name = args.model.replace(":","_")
    output_subdir = f"output_{model_name}_wo_RAG"
    output_dir = os.path.join(BASE_DIR, output_subdir)
    cache_file = os.path.join(BASE_DIR, output_subdir, "task_progress.txt")
    log_file = os.path.join(BASE_DIR, output_subdir, "task_log.csv")

    print("Initializing BIRD Metadata Cache...")
    meta_cache = BirdMetaCache(BASE_DIR)

    print(f"Initializing AutoCodingAgent with model: {args.model}")
    agent = AutoCodingAgent(model_name=args.model) 
    
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("question_id,status,attempts\n")

    tasks = []

    with open(TASK_FILE, 'r', encoding='utf-8') as f:
        content = json.load(f)
        if isinstance(content, list):
            tasks = content

    
    last_success_id = None
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            last_success_id = f.read().strip()
            print(f"Resuming from after task: {last_success_id}")

    skipping = True if last_success_id else False

    for task in tqdm(tasks):
        current_id = str(task['question_id'])
        if skipping:
            if current_id == str(last_success_id):
                skipping = False
            continue

        status, attempts = run_task(agent, task, meta_cache, output_dir)
        print(f"Task {current_id} finished with status: {status}, attempts: {attempts}")

        with open(log_file, 'a') as f:
            f.write(f"{current_id},{status},{attempts}\n")

        with open(cache_file, 'w') as f:
            f.write(current_id)
    
if __name__ == "__main__":
    main()



