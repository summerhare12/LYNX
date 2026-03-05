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

class BirdMetaCache:
    def __init__(self, bird_dir):
        self.bird_dir = bird_dir
        self.tables_path = os.path.join(bird_dir, "dev_tables.json")
        self.schemas = {} # 缓存: {db_id: schema_info_dict}
        self._load_all_schemas()

    def _load_all_schemas(self):
        """一次性加载 dev_tables.json 并解析所有 DB 的 Schema"""
        if not os.path.exists(self.tables_path):
            print(f"[Error] Schema file not found: {self.tables_path}")
            return

        print(f"Loading BIRD schemas from {self.tables_path}...")
        try:
            with open(self.tables_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将 list 转为以 db_id 为 key 的字典，方便后续查找
            for db_item in tqdm(data, desc="Caching schemas"):
                db_id = db_item["db_id"]
                self.schemas[db_id] = self._parse_schema_item(db_item)
                
        except Exception as e:
            print(f"[Error] Failed to load dev_tables.json: {e}")

    def _parse_schema_item(self, db_schema):
        """解析单个数据库的 Schema 信息为结构化数据"""
        table_names_original = db_schema['table_names_original']
        column_names_original = db_schema['column_names_original']
        column_types = db_schema['column_types']
        primary_keys = db_schema['primary_keys']
        foreign_keys = db_schema['foreign_keys']

        # 1. 预处理表结构文本
        schema_text = []
        schema_text.append(f"Database Schema for {db_schema['db_id']}:\n")

        tables_struct = {i: {"name": name, "columns": []} for i, name in enumerate(table_names_original)}
        
        # 字段列表，用于后续精确匹配检查
        column_list_json = {t: [] for t in table_names_original}

        for col_idx, (table_idx, col_name) in enumerate(column_names_original):
            if table_idx == -1: continue
            
            table_name = table_names_original[table_idx]
            col_type = column_types[col_idx]
            is_pk = col_idx in primary_keys
            
            col_desc = f"{col_name} ({col_type})"
            if is_pk:
                col_desc += " [PK]"
            
            tables_struct[table_idx]["columns"].append(col_desc)
            column_list_json[table_name].append(col_name)

        for _, info in tables_struct.items():
            schema_text.append(f"Table `{info['name']}`:")
            schema_text.append(f"  Columns: {', '.join(info['columns'])}")
        
        if foreign_keys:
            schema_text.append("\nForeign Keys:")
            for (src_col_idx, tgt_col_idx) in foreign_keys:
                src_info = column_names_original[src_col_idx]
                tgt_info = column_names_original[tgt_col_idx]
                src_tbl = table_names_original[src_info[0]]
                tgt_tbl = table_names_original[tgt_info[0]]
                schema_text.append(f"  {src_tbl}.{src_info[1]} -> {tgt_tbl}.{tgt_info[1]}")
        
        return {
            "schema_text": "\n".join(schema_text),
            "table_names": table_names_original,
            "column_json": json.dumps(column_list_json, indent=2)
        }

    def get_database_meta(self, db_id):
        """
        获取指定 db_id 的完整元数据（Schema + CSV descriptions）
        Schema 从内存取，CSV 从磁盘读并清洗
        """
        if db_id not in self.schemas:
            return f"Database {db_id} not found in cache.", [], "{}"
        
        # 1. 获取内存中的 Schema
        cached_data = self.schemas[db_id]
        final_summary = [cached_data["schema_text"]]

        # 2. 实时读取 CSV 描述文件并清洗
        desc_dir = os.path.join(self.bird_dir, "dev_databases", db_id, "database_description")
        if os.path.exists(desc_dir):
            csv_files = glob.glob(os.path.join(desc_dir, "*.csv"))
            if csv_files:
                final_summary.append("\nDetailed Column Descriptions:\n")
                for csv_file in csv_files:
                    filename = os.path.basename(csv_file)
                    table_name = filename.replace(".csv", "") # 假设文件名对应表名
                    
                    try:
                        try:
                            df = pd.read_csv(csv_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(csv_file, encoding='cp1252')
                        
                        df.columns = df.columns.str.lower()
                        
                        target_cols = ['original_column_name', 'column_description', 'value_description']
                        available_cols = [c for c in target_cols if c in df.columns]
                        
                        if not available_cols:
                            continue

                        df = df[available_cols].dropna(how='all') # 去掉全空的行
                        
                        if df.empty:
                            continue

                        description_lines = []
                        description_lines.append(f"Table `{table_name}` descriptions:")
                        
                        for _, row in df.iterrows():
                            col_name = row.get('original_column_name')
                            if pd.isna(col_name): continue
                            
                            desc = row.get('column_description')
                            vals = row.get('value_description')
                            
                            # 构建描述字符串
                            line_parts = [f"  - {col_name}:"]
                            if pd.notna(desc) and str(desc).strip():
                                line_parts.append(f" {str(desc).strip()}")
                            if pd.notna(vals) and str(vals).strip() and str(vals).strip().lower() != "nan":
                                # 清洗 values，去掉过长的 commonsense
                                val_str = str(vals).strip()
                                if "NOT USEFUL" in val_str: continue # 跳过无用列
                                line_parts.append(f" [Values: {val_str}]")
                            
                            # 只有当 desc 或 vals 有意义时才添加
                            if len(line_parts) > 1:
                                description_lines.append("".join(line_parts))
                        
                        if len(description_lines) > 1: # 如果只有表名那行，就不加了
                            final_summary.append("\n".join(description_lines) + "\n")
                            
                    except Exception as e:
                        print(f"[Warn] Failed to read/parse {filename}: {e}")
        
        return "\n".join(final_summary), cached_data["table_names"], cached_data["column_json"]


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
            return match.group(1)
        if "import " in text or "print(" in text:
            return text
        return None


