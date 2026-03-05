import os
import re
import sys
import subprocess
import time
import shutil
from typing import List, Tuple, Optional
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from tools import *
from bqtools import *
from embedding import create_vector_store, default_embeddings

class AutoCodingAgent:
    def __init__(self, model_name: str = "deepseek-chat", data_files: Optional[List[str]] = None):
        if "qwen" in model_name:
            self.llm = OllamaLLM(model=model_name, temperature=0)
        elif "deepseek" in model_name:
            api_key = os.getenv() 
            
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com", 
                temperature=0
            )
        elif "gpt" in model_name:
            # key from your code snippet
            api_key = ""
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=0
            )

        self.data_files = data_files  
        self.query = ""   
        self.response = ""    
        self.error_history = ""
        self.all_table_names = []
        self.column_names = ""
        self.last_code = ""
        self.last_error = ""
    
    def extract_terms(self):
        extraction_prompt = PromptTemplate.from_template(
        """Please select important specialized terms in the task question, which you do not understand. 
        **Output**: Only return a comma-separated **list** of terms. NO EXTRA WORDS. If none, return 'None'.
        Task question: {query}.
        """)
        chain = extraction_prompt | self.llm | StrOutputParser()
        terms_str = chain.invoke({"query": self.query})
        terms_str = re.sub(r'<think>.*?</think>', '', terms_str, flags=re.DOTALL).strip()
        
        if "None" in terms_str or not terms_str.strip():
            return []
            
        terms = [t.strip() for t in terms_str.split(",") if t.strip()]
        return terms


    def get_knowledge(self):
        print(f"Analyzing the query for key terms...")
        terms = self.extract_terms()

        if not terms:
            print("No key terms found.")
            return ""
            
        try:
            docs = load_wiki(terms)
            if not docs:
                return ""
            
            split_docs = split_documents(docs)
            vectorstore = create_vector_store(split_docs)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.invoke(self.query)
            knowledge = "\n\n".join([f"\n{d.page_content}" for d in relevant_docs])

            return knowledge

        except Exception as e:
            print(f"[!] Wiki 查询失败: {e}")
            return ""


    def original(self, context: str, data_context: str) -> str:
        if len(data_context) > 350000:
            data_context = data_context[:350000] + "\n...(truncated due to length limit)..."
        prompt_template = '''
        You are an excellent Bigquery SQL expert.
        Your task is to write a SQL query to answer the user's request based on the provided database schema.

        User request: {query}
        Data Overview:{data_context}
        Relevant knowledge:{context}
         **Step-by-Step Reasoning Strategy**:
        1. **Deconstruct the Request**: Break down the user's question into specific filters, aggregations, and time windows. Note any specific exclusions or conditions (e.g., "no positive engagement", "7-day period").
        2. **Consult Knowledge**: Check the "Relevant Knowledge" section. Refer to useful explanations of specific terms (e.g., "pseudo users", "engagement time").
        3. **Locate Schema Elements**: Identify the exact table names (including project and dataset) and column names from "Data Overview". Do not hallucinate columns.
        4. **Formulate the Query**:
           - Use the correct tables (use Fully Qualified Names like `project.dataset.table`).
           - Apply clauses for all constraints.
           - Handle date/time parsing if necessary.
           - Enclose column names and string literals in double quotes `"`. Mind case sensitivity.
        5. **Final Review**: Does the SQL answer the exact question asked?

        **Output Requirements**:
        1. First, provide a short Analysis of why the error happened and how you will fix it.
        2. Then, output the Corrected SQL inside a Markdown code block.

        ** Output Format**:
        
        Analysis: [Your brief debugging thought process]

        ```sql
        SELECT (your SQL clause)
        FROM {fqn}
        (your SQL clause)
        ```
        (Replace with the actual table and column names using Fully Qualified Name)
        '''
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()

        fqn = ""
        for table_name in self.all_table_names:
            if "Table groups" not in table_name:
                fqn = table_name
        try:
            response = chain.invoke({
                "query": self.query, 
                "context": context, 
                "data_context": data_context,
                "fqn": fqn,
            })
        except Exception as e:
            if "Content Exists Risk" in str(e) or "content policy" in str(e).lower():
                print(f"\n[Warn] API Content Filter triggered. Retrying without external knowledge context...")
                response = chain.invoke({
                    "query": self.query, 
                    "context": "External knowledge omitted due to content filter risk.", 
                    "data_context": data_context,
                    "fqn": fqn,
                })
            else:
                raise e

        return extract_code(response)

    def revise(self, error: str, reference: str, exe_flag: bool) -> str:
        # if exe_flag is False:
        if len(self.error_history) > 350000:
            self.error_history = self.error_history[:350000] + "\n...(truncated due to length limit)..."
        prompt_template = """
        You are an excellent Bigquery SQL expert.
        The previous SQL query failed to execute. Your task is to correct the SQL based on the error message.
        
        **Context**:
        - User request: {query}
        - latest executed SQL:
        {SQL}
        - Error:
        {error}
        - Error history:
        {error_history}
        - Reference / Hints:
        {reference}

        **Step-by-Step Debugging Strategy (CoT)**:
        1. **Analyze the Error**: Read the error message carefully. It's a syntax error, schema mismatch or a logic issue?
        2. **Locate the Fault**: Identify the exact line or clause in the SQL that caused the error.
            - If it's a syntax error, check the command and function usage.
            - If it's schema mismatch, check the correct FQN for tables and correct case-sensitive columns enclosed in double quotes.
            - If it's a logic issue, review the user request constraints (e.g. date ranges, filters) carefully.
        3. **Formulate a Fix**: Make use of your knowledge and the reference/hints to fix the issues.
        4. **Final Refinement**: Ensure the rest of the query (filters, logic) remains consistent with the User Request.
            - For syntax/schema: applying the correct function or name.
            - For logic: consider rewriting the CTEs or subqueries to align with the correct logic.
        **Output Requirements**:
        1. First, provide a short Analysis of why the error happened and how you will fix it.
        2. Then, output the Corrected SQL inside a Markdown code block.

        ** Output Format**:
        
        Analysis: [Your brief debugging thought process]

        ```sql
        SELECT (your SQL clause)
        FROM {fqn}
        (your SQL clause)
        ```
        (Replace with the actual table and column names using FQN)
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        fqn = ""
        for table_name in self.all_table_names:
            if "Table groups" not in table_name:
                fqn = table_name
                
        response = chain.invoke({
            "query": self.query, 
            "SQL": self.response,
            "error": error,
            "error_history": self.error_history,
            "reference": reference,
            "fqn":fqn,
        })
        
        # else:
        #     prompt_template = """
        #     Here is a SQL query that executed successfully. I need you to verify if it logically matches the user's question.

        #     User Question: {query}
        #     Candidate SQL: {previous_code}
        #     Error history: {error_history}

        #      **Reasoning Checklist**:
        #     1. **Explain the SQL**: Describe what the Candidate SQL does in plain English. (e.g., "It selects users who visited page X, showing only those active in the last 7 days.")
        #     2. **Compare with Request**: Check whether each part of your explanation exactly matches the the specific constraints in the User Question.
        #     3. **Decision**:
        #        - If the logic is FLAWED: Write a corrected SQL query. Be aware to use Fully Qualified Name for tables and enclose columns in double quotes. Refer to error history to prevent repeating mistakes.
        #        - If the logic is PERFECT: Output the original SQL unchanged.

        #     **Output Format**:
            
        #     Explanation: [Your plain English translation of the SQL]
        #     Analysis: [Your comparison logic]

        #     ```sql
        #     (The Final SQL Query)
        #     ``` 
        #     """
            

        #     prompt = PromptTemplate.from_template(prompt_template)
        #     chain = prompt | self.llm | StrOutputParser()
        #     response = chain.invoke({
        #         "query": self.query, 
        #         "previous_code": self.response,
        #         "error_history": self.error_history,
        #     })
        
        return extract_code(response)
    

    def reflect(self, error: str):
        history = self.error_history
        if len(history) > 350000:
            history = history[:350000] + "\n...(truncated due to length limit)..."
        prompt_template = """
        You are an expert Reviewer. Your goal is to check the debugging progress and provide actionable suggestions for the next step.

        **Context**:
        1. **Previous Error History**: 
        {history}
        2. **Last Encountered Error**: 
        {last_error}
        3. **Last SQL Code**
        {last_code}
        4. **Current Error**: 
        {current_error}
        5. **Current SQL Code**:
        {current_code}

        **Reflection Process (Chain of Thought)**:
        1. **Compare Errors**: Did the "Last Encountered Error" disappear in the "Current Error"? 
        - If YES: The previous fix was successful (or at least moved past that bug).
        - If NO (Same Error): The previous fix FAILED. We are stuck in a loop.
        2. **Formulate Advice**: 
        - If progressing: Summarize the error. Compare the last SQL code with the current SQL code, and describe the solution to it.
        - If stuck: Suggest a different approach. You can also refer to the previous error history for some inspirations.

        **Output Format:**
        Current Code [{current_code}]. Current Error[{current_error}].
        Previous fix [Worked/Failed]. Suggestion: [Action].
        """
        # prompt_template = """
        # You are an expert Reviewer. Your goal is to summarize the debugging progress and provide actionable suggestions for the next step.

        # **Context**:
        # 1. **Previous Error History (Summarized)**: 
        # {history}
        # 2. **Last Encountered Error**: 
        # {last_error}
        # 3. **Last SQL Code**
        # {last_code}
        # 4. **Current Error**: 
        # {current_error}
        # 5. **Current SQL Code**:
        # {current_code}

        # **Reflection Process (Chain of Thought)**:
        # 1. **Compare Errors**: Did the "Last Encountered Error" disappear in the "Current Error"? 
        #    - If YES: The previous fix was successful (or at least moved past that bug).
        #    - If NO (Same Error): The previous fix FAILED. We are stuck in a loop.
        # 2. **Formulate Advice**: 
        #    - If progressing: Summarize the error. Compare the last SQL code with the current SQL code, and describe the solution to it.
        #    - If stuck: Suggest a different approach. You can also refer to the previous error history for some inspirations.

        # **Output Format:**(if Current Error is "No error", only output Error History part)
        # Error History:
        # 1. Error: [Summarized Contents]. Solution: [Summarized Contents].
        # 2. Error: [Summarized Contents]. Solution: [Summarized Contents].
        # ...
        # Latest blocker:
        # Previous fix [Worked/Failed]. Current blocker is [Error]. Suggestion: [Action].
        # """
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "history": history,
            "last_error": self.last_error,
            "last_code": self.last_code,
            "current_error": error,
            "current_code": self.response,
            })
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        if "no previous errors" in response.lower() or "summary is an empty string" in response.lower():
            return ""

        print(response)

        return response

    def run(self, query: str, knowledge_path: str, data_paths: list, attempt: int, error: str, exe_flag: bool):  
        reference = ""
        if attempt == 1:
            self.query = query   
            self.error_history = ""
            self.last_code = ""
            self.last_error = ""
            context = ""
            data_context = ""
            all_summaries = []
            table_names = []
            all_column_names = {}
            schema = ""
            # if knowledge_path:
            #     try:
            #         with open(knowledge_path, 'r', encoding='utf-8') as f:
            #             context += f"\nexternal knowledge from document: {f.read()}\n\n"
            #     except Exception as e:
            #         print(f"[Warn] Failed to read external knowledge: {e}")
            
            # context += "--- knowledge for related terms from wikipedia(only for reference) ---\n"
            # context += self.get_knowledge()

            # for data_path in data_paths:
            #     schema = os.path.basename(os.path.normpath(data_path))

            #     data_summary, table_name, column_name = load_metadata(data_path)
            #     all_summaries.append(data_summary)
                
            #     table_names.extend(table_name)

            #     column_json = json.loads(column_name)
            #     all_column_names.update(column_json)
                     
            # data_context = "\n".join(all_summaries)
            # self.all_table_names = table_names
            # # data_context += f"Table FQN: {self.all_table_names}"
            # self.column_names = json.dumps(all_column_names, indent=2, ensure_ascii=False)            
            self.response = self.original(context, data_context)

        else:
        # elif exe_flag is False:
            # self.error_history = self.reflect(error)
            self.error_history += "\n" + self.reflect(error)
            self.last_code = self.response
            self.last_error = error
            if "database" in error or "not exist" in error or "Access Denied" in error or re.search(r"Dataset .*? was not found", error):
                str_table_names = "\n".join(self.all_table_names)
                reference = f"Please use the correct Fully Qualified Name after FROM. {str_table_names} "
            elif "Unrecognized name" in error or "identifier" in error or re.search(r"Name .*? not found", error):
                if len(self.column_names) > 350000:
                    self.column_names = self.column_names[:350000] + "\n...(truncated due to length limit)..."
                reference = f"""Please use the correct column names. 
                **Important**: Bigquery columns are case-sensitive. You MUST enclose the column name in **double quotes** (e.g., "publication_date") and pay attention to case sensitivity.
                Available columns: {self.column_names}"""
            elif "Query executed successfully but returned no data" in error:
                str_table_names = "\n".join(self.all_table_names)
                reference = f"""  
                Schema information for reference: {str_table_names}              
                **Debugging Strategy**:
                1. **Contradictory Conditions**: Did you use `WHERE column = 'A' AND column != 'A'`?
                   - *Fix*: Use a **subquery** or **CTE** to first find the users/IDs who match condition 'A', and THEN select their other actions where `column != 'A'`.
                2. **Check String Literals**: Are you using the exact case-sensitive name? (e.g., "YouTube" vs "Youtube").
                3. **Check Date Formats**: Ensure dates match the database format (e.g., '20170701' vs '2017-07-01').
                4. **JOINs**: Check if your JOIN conditions are causing data loss.
                """
            elif "timeout" in error:
                reference = """
                **Performance Optimization Required**: The query exceeded the time limit.
                1. Avoid `ARRAY_CONTAINS` in JOINs or WHERE clauses if possible: It is very slow on large datasets. Try to FLATTEN the array first (`LATERAL FLATTEN`).
                2. Reduce scanned data: Apply strict date pre-filters or other indexed column filters *before* joining.
                3. Use CTEs wisely: Materialize small intermediate results, but be careful with large ones.
                4. Simplify Logical Checks: Instead of `EXISTS (SELECT ... LIKE 'A61%')`, consider flattening the CPC array and directly filtering on the flattened column.
                """
            else:
                reference = search_error(error)
            # code_error = f"\nlatest code:\n{self.response}\n\n meets problem:\n{error}\n"
            self.response = self.revise(error, reference, exe_flag)
        # else:
        #     self.error_history = self.reflect("No error.")
        #     self.last_code = self.response
        #     self.last_error = "Execution Successful"
        #     self.response = self.revise("", reference, exe_flag)

        print(self.response)
        return self.response
        