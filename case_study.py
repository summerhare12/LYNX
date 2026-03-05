import os
import re
import pandas as pd
import json
import argparse
from casestudy_agent import AutoCodingAgent
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from google.oauth2 import service_account
from google.cloud import bigquery

BASE_DIR = "./Spider2/spider2-lite"
DATA_DIR = os.path.join(BASE_DIR, "resource/databases/bigquery")
TASK_FILE = "./Spider2/spider2-lite/spider2-lite.jsonl"
CREDENTIAL_PATH = './Spider2/spider2-lite/bigquery_key.json' 


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

def external_knowledge_path(instance_id, knowledge_filename):
    if not knowledge_filename:
        return None
        
    path = os.path.join(BASE_DIR, "resource/documents",knowledge_filename)
    if os.path.exists(path):
        return path
    else:
        return None

def run_task(agent, task, client):
    instance_id = task['instance_id']
    instruction = task['question']
    db_id = task['db']
    knowledge_file = task['external_knowledge']
    error = ""
    sql_query = ""
    exe_flag = False
    
    print(f"\n=== Running Task: {instance_id} ===")
    print(f"Query: {instruction}")
    
    metadata_dir = get_metadata_path(instance_id, db_id)
    knowledge_path = external_knowledge_path(instance_id, knowledge_file)
    # print(f"metadata_dir: {metadata_dir}, external_knowledge_path: {knowledge_path}\n")
    if not metadata_dir:
        return "Metadata Not Found", 0

    for attempt in range(1, MAX_TRIES + 1):
        print(f"\n--- attempt {attempt} ---\n")
        # sql_query = agent.run(instruction, knowledge_path, metadata_dir, attempt, error, exe_flag)
        sql_query="""
        -- Highest-volume active council district for intra-district trips (different stations)
        WITH stations AS (
        SELECT
            stationid,
            councildistrict,
            status
        FROM `bigquery-public-data.austinbikeshare.bikesharestations`
        WHERE LOWER(status) = 'active'
        ),
        trips AS (
        SELECT
            CAST(startstationid AS INT64) AS start_id,
            CAST(endstationid AS INT64) AS end_id
        FROM `bigquery-public-data.austinbikeshare.bikesharetrips`
        WHERE startstationid IS NOT NULL
            AND endstationid IS NOT NULL
        )

        SELECT
        s.councildistrict AS district,
        COUNT(*) AS trip_count
        FROM trips t
        JOIN stations s ON t.start_id = s.stationid          -- start station (active) with district
        JOIN stations e ON t.end_id = e.stationid            -- end station (active) with district
        WHERE s.councildistrict IS NOT NULL
        AND e.councildistrict IS NOT NULL
        AND s.councildistrict = e.councildistrict          -- same district
        AND t.start_id <> t.end_id                         -- different stations
        GROUP BY district
        ORDER BY trip_count DESC
        LIMIT 1;
        """
        if not sql_query:
            error = "Query is empty. Please write a single valid Bigquery SQL query to retrieve the right answer."
            print(error)
            continue
        try:
            query_job = client.query(sql_query)  # API request
            df = query_job.result().to_dataframe()            
            
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
            #     error = ""
            #     exe_flag = True
            #     continue
            else:
                # output_path = os.path.join(output_dir, f"{instance_id}.csv")
                # output_dir = os.path.dirname(output_path)
                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)
                # df.to_csv(output_path, index=False)
                # print(f"Results saved to {output_path}")

                # output_sql = os.path.join(output_dir, f"{instance_id}.txt")
                # with open(output_sql, 'w', encoding='utf-8') as f:
                #     f.write(sql_query)
            
                return "Success!", attempt
        
        except FunctionTimedOut:
            print(f"[Error] SQL execution timed out!")
            error = "SQL execution timed out!"
            exe_flag = False

        except Exception as e:
            print(f"BigQuery SQL Error: {e}")
            error = str(e)
            exe_flag = False
    
    # output_path = os.path.join(output_dir, f"{instance_id}.csv")
    # output_dir = os.path.dirname(output_path)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # pd.DataFrame().to_csv(output_path, index=False)
    # print(f"Failed after{MAX_TRIES} attempts. Empty CSV saved to {output_path}")

    # if sql_query:
    #     output_sql = os.path.join(output_dir, f"{instance_id}.txt")
    #     with open(output_sql, 'w', encoding='utf-8') as f:
    #         f.write(sql_query)
        
    return "Fail.", MAX_TRIES



def main(): 
    agent = AutoCodingAgent()  
    credentials = service_account.Credentials.from_service_account_file(CREDENTIAL_PATH)
    client = bigquery.Client(credentials=credentials)
   
    task = {"instance_id": "bq282", "db": "austin", "question": "Can you tell me the numeric value of the active council district in Austin which has the highest number of bike trips that start and end within the same district, but not at the same station?", "external_knowledge": None}


    status, attempts = run_task(agent, task, client)   
if __name__ == "__main__":
    main()



