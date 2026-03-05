import os
import re
import io
import ast
import time
import glob
import json
from tqdm import tqdm
import pandas as pd
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embedding import default_embeddings


def split_documents(docs, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n",  ".", ",", "?", "!", " ", "", "。", "，", "、", "！", "？"], 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = []
    for doc in tqdm(docs, desc="Splitting documents"):
        try:
            parts = text_splitter.split_documents([doc])
            split_docs.extend(parts)
        except Exception as e:
            tqdm.write(f"split failed for doc (metadata={getattr(doc, 'metadata', None)}): {e}")
    return split_docs


def load_wiki(term_list, lang="zh"):
    """
    check Wikipedia for the term list
    """
    all_wiki_docs = []
    
    print(f"begin {len(term_list)} terms")
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    
    for term in tqdm(term_list, desc="Loading Wikipedia Articles"):
        try:
            loader = WikipediaLoader(query=term, lang=lang, load_max_docs=1, doc_content_chars_max=3000) 

            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_type"] = "wikipedia"
                doc.metadata["search_term"] = term
                
            all_wiki_docs.extend(docs)
            print(f"succeed: {term}")
            time.sleep(1)  # 避免请求过快
            
        except Exception as e:
            print(f"fail {term}: {e}")
            
    return all_wiki_docs

def search_web(query: str, max_results: int = 3, max_chars: int = 900):
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results, region="wt-wt")
    try:
        print(f"[*] Searching web for: {query}")
        results = wrapper.results(query, max_results=max_results)
        
        if not results:
            return ""
            
        formatted_results = []
        
        for i, res in enumerate(results, 1):
            title = res.get('title', 'No Title')
            snippet = res.get('snippet', 'No Snippet')
            
            entry = f"# Result {i}: {title}\nContent: {snippet}\n"
            formatted_results.append(entry)

        final_output = "\n".join(formatted_results)
        if len(final_output) > max_chars:
            final_output = final_output[:max_chars]
            
        return final_output

    except Exception as e:
        print(f"[!] Web search failed: {e}")
        return ""


def normalize_table_name(name):
    """
    将表名中的连续数字替换为占位符，用于识别同构表。
    例如: GA_SESSIONS_20170801 -> GA_SESSIONS_{NUM}
    """
    return re.sub(r'\d+', '{NUM}', name)

def load_metadata(metadata_dir: str): # 核心函数
    table_names = []
    column_names = {}
    summary = []
    
    if not os.path.exists(metadata_dir):
        return f"Error: Metadata directory not found: {metadata_dir}"

    ddl_path = os.path.join(metadata_dir, "DDL.csv")
    table_groups = {}

    if os.path.exists(ddl_path):
        try:
            df = pd.read_csv(ddl_path)
            # df = pd.read_csv(ddl_path, index_col='table_name')
            df.columns = df.columns.str.lower()
            if 'table_name' in df.columns:
                df = df.set_index('table_name')
            
            total_number = len(df)
            summary.append(f"The schema has {total_number} tables in total.\n")
            
            for index, row in df.iterrows():
                norm_name = normalize_table_name(index)
                if norm_name not in table_groups:
                    table_groups[norm_name] = []
                table_groups[norm_name].append(index) #同构表分组

            # print(table_groups)
        
        except Exception as e:
            print(f"[Warning] Failed to parse DDL.csv: {e}")
            return f"Error parsing DDL: {e}",[] , "{}"

    for norm_name, indexes in table_groups.items():
        first_table_name = indexes[0]
        # desc = df.loc[first_table_name, 'description']
        ddl = df.loc[first_table_name, 'ddl']

        # database = "UnknownDB"
        # schema = "UnknownSchema"

        for index in indexes:
            json_name = index
            if "." in json_name:
                json_name = json_name.split(".")[-1]
            json_file = glob.glob(os.path.join(metadata_dir, "**", f"*{json_name}.json"), recursive=True)
            if json_file:
                target_json = json_file[0]
                try:
                    with open(target_json, 'r') as f:
                        data = json.load(f)
                    table_fullname = data.get('table_fullname')
                    columns = data.get('column_names', [])
                    descriptions = data.get('description', [])
                    sample_rows = data.get('sample_rows', [])
                    break
                except Exception as e:
                    print(f"[Warning] Failed to read JSON {target_json}: {e}")
                    continue
            else:
                continue
        if not json_file:
            return "Json Metadata not found.", [], "{}"
               
        count = len(indexes)
        if count > 1:
            last_table_name = indexes[-1]
            table_names.append(f"Table groups {norm_name} has {count} tables, from the first table '{first_table_name}' to the last '{last_table_name}'.")
            column_names[norm_name] = ddl
            summary.append(
            f"""Table groups {norm_name} has {count} tables, from the first table '{first_table_name}' to the last '{last_table_name}'. They have the same structure.\n
            Data Definition Language:\n{ddl}\n""")
            # if pd.notna(desc) and desc:
            #     summary.append(f"Description: {desc}\n")
        else:
            table_names.append(f"{first_table_name}")
            column_names[first_table_name] = ddl
            summary.append(
            f"""Data Definition Language of the table '{first_table_name}':\n{ddl}\n""")
            # if pd.notna(desc) and desc:
            #     summary.append(f"Description: {desc}\n")

        summary.append("---------------------------\n")   

    return "".join(summary), table_names, json.dumps(column_names, indent=2, ensure_ascii=False)


def try_parse_json(value):
    if isinstance(value, str):
        if (value.strip().startswith('{') and value.strip().endswith('}')) or \
           (value.strip().startswith('[') and value.strip().endswith(']')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


def clean(data):
    """
    递归清洗 JSON 数据，对过长的字符串和列表进行截断。
    """
    if isinstance(data, dict):
        return {k: clean(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        if len(data) > 1:
            truncated_list = [clean(data[0])]
            truncated_list.append("... (truncated)")
            return truncated_list
        else:
            return [clean(item) for item in data]
            
    elif isinstance(data, str):
        # 尝试解析嵌套的 JSON 字符串 (例如 '{"a": 1}')
        if (data.strip().startswith('{') and data.strip().endswith('}')) or \
           (data.strip().startswith('[') and data.strip().endswith(']')):
            try:
                parsed = json.loads(data)
                return clean(parsed)
            except json.JSONDecodeError:
                pass 

        if len(data) > 15:
            return data[:15] + "... (truncated)"
        return data
        
    else:
        return data


def extract_code(text: str):
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
            clean_code = re.sub(r'--.*', '', code)
            clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
            
            if not clean_code.strip():
                return None
            
            return code
        if "import " in text or "print(" in text:
            return text
        return None


