import os
import re
import sys
import logging
import subprocess
import time
import shutil
from typing import List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from birdtools import *
from embedding import create_vector_store, default_embeddings


class AutoCodingAgent:
    def __init__(self, model_name):
        api_key = os.getenv("zju-aiops", "sk-2da937ce5f424f57b35058e4c277488d") 
        
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com", 
            temperature=0
        ) 
        self.query = ""   
        self.response = ""    
        self.error_history = ""
        self.database = ""
        self.schema_info = ""
        self.few_shot_examples = """
        Example 1:
        question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?
        evidence: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`,
        SQL: SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1

        Example 2:
        question: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
        evidence: Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`
        SQL: SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3

        Example 3: 
        question: Name chemical elements that form a bond TR001_10_11.
        evidence: element = 'cl' means Chlorine; element = 'c' means Carbon; element = 'h' means Hydrogen; element = 'o' means Oxygen, element = 's' means Sulfur; element = 'n' means Nitrogen, element = 'p' means Phosphorus, element = 'na' means Sodium, element = 'br' means Bromine, element = 'f' means Fluorine; element = 'i' means Iodine; element = 'sn' means Tin; element = 'pb' means Lead; element = 'te' means Tellurium; element = 'ca' means Calcium; TR001_10_11 is the bond id; molecule id refers to SUBSTR(bond_id, 1, 5); atom 1 refers to SUBSTR(bond_id, 7, 2); atom 2 refers to SUBSTR(bond_id, 10, 2)
        SQL: SELECT T1.element FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id INNER JOIN bond AS T3 ON T2.bond_id = T3.bond_id WHERE T3.bond_id = 'TR001_10_11'
        """
    
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
            print(f"[!] Wiki ��ѯʧ��: {e}")
            return ""


    def original(self, context: str, data_context: str) -> str:
        if len(data_context) > 350000:
            data_context = data_context[:350000] + "\n...(truncated due to length limit)..."
        prompt_template = '''
        You are an excellent SQlite SQL expert.
        Your task is to write a **correct and efficient** SQL query to answer the user's request based on the provided database schema.

         **BIRD Benchmark Survival Guide (CRITICAL)**:
        1. **Fuzzy Matching**: User terms often don't match database values exactly (e.g., 'US' vs 'USA'). Refer to data overview and properly use `LIKE`. 
        2. **Date Chaos**: Dates might be 'YYYY-MM-DD', 'YYYY/MM/DD', or even 'YYYYMMDD' strings. Use `SUBSTR` or `STRFTIME` to standardize before comparing.
        3. **Case Sensitivity**: SQLite is case-sensitive for `=`. Always consider `LOWER(column) = 'value'`.
        4. **Join Paths**: Look for the shortest, most logical join path. Avoid circular joins.
        5. **No Defensive Logic**: **NEVER** write `CASE WHEN COUNT(*) = 0 THEN 'No data'`. Just return the empty set if no data matches.

        Here is some examples:
        {few_shots}

        User request: {query}
        Data Overview:{data_context}
        Relevant knowledge:{context}

        **Step-by-Step Reasoning Strategy**:
        1. **Deconstruct the Request**: Break down the user's question into specific filters, aggregations, and time windows.
        2. **Schema Mapping**: Find the exact tables and columns according to data overview. **Check for "ID" columns vs "Name" columns**.
        3. **Logic Formulation**: Construct the query logic.
        4. **Final Polish**: Check for common pitfalls listed in the Survival Guide above.

        **Output Requirements**:
        1. First, provide a short Analysis.
        2. Then, output the SQL inside a Markdown code block.

        ** Output Format**:
        
        Analysis: [Your brief debugging thought process]

        ```sql
        SELECT ...
        ```
        '''
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        try:
            response = chain.invoke({
                "few_shots": self.few_shot_examples,
                "query": self.query, 
                "context": context, 
                "data_context": data_context,
            })
        except Exception as e:
            if "Content Exists Risk" in str(e) or "content policy" in str(e).lower():
                print(f"\n[Warn] API Content Filter triggered. Retrying without external knowledge context...")
                response = chain.invoke({
                    "few_shots": self.few_shot_examples,
                    "query": self.query, 
                    "context": "External knowledge omitted due to content filter risk.", 
                    "data_context": data_context,
                })
            else:
                raise e

        return extract_code(response)

    def revise(self, error: str, reference: str, exe_flag: bool) -> str:
        if exe_flag is False:
            if len(self.error_history) > 350000:
                self.error_history = self.error_history[:350000] + "\n...(truncated due to length limit)..."
            prompt_template = """
            You are an excellent SQlite SQL expert.
            Here is some examples of text-to-SQL:
            {few_shots}

            The previous SQL query failed to execute. Your task is to correct the SQL based on the error message.
            
                **Context**:
            - User request: {query}
            - Schema Information: {schema_info}
            - latest executed SQL: {code}
            - Error: {error}
            - Error history:
            {error_history}
            - Reference / Hints:
            {reference}

            **Step-by-Step Debugging Strategy (CoT)**:
            1. **Check Inspection Results**: Look at the `error_history` above. Did we run an `INSPECT` query? usage that info to fix schema/value mismatches.
            2. **Analyze the Error**: Read the error message carefully. It's a syntax error, schema mismatch or a logic issue?
            3. **Locate the Fault**: Identify the exact line or clause in the SQL that caused the error.
            4. **Formulate a Fix**: Make use of your knowledge and the reference/hints to fix the issues.
                - Do NOT assume 'YYYY-MM-DD' or 'YYYYMMDD'. Use `STRFTIME` or `SUBSTR` carefully. 
                - Prefer `STRFTIME('%Y', col)` for 'YYYY-MM-DD'. If the format is 'YYYYMMDD', use `SUBSTR(col, 1, 4)`.
                - Use `JULIANDAY(end) - JULIANDAY(start)` for day differences.
                - SQLite `LIKE` is case-insensitive for ASCII, but `=` is case-sensitive. Use `LOWER(col) = 'value'` if unsure.
                - Do not use `LIKE` when looking for specific IDs or Status codes unless searching for a pattern.
                - **No Defensive Logic**: **NEVER** write `CASE WHEN COUNT(*) = 0 THEN 'No data'`. Just return the empty set if no data matches.
                - For ratios/percentages, always multiply by `1.0` or `100.0` to force floating-point division (e.g., `cnt * 100.0 / total`).
            5. **Final Refinement**: Ensure the rest of the query (filters, logic) remains consistent with the User Request.

            **Output Requirements**:
            1. First, provide a short Analysis of why the error happened and how you will fix it.
            2. Then, output the Corrected SQL inside a Markdown code block.

            ** Output Format**:
            
            Analysis: [Your brief debugging thought process]

            ```sql
            SELECT (your SQL clause)
            FROM (the exact name)
            (your SQL clause)
            ```
            """
            
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
                    
            response = chain.invoke({
                "few_shots": self.few_shot_examples,
                "query": self.query, 
                "schema_info": self.schema_info,
                "code": self.response,
                "error": error,
                "error_history": self.error_history,
                "reference": reference,
            })
            
        else:
            prompt_template = """
            Here is a SQL query that executed successfully. 
            Your task is to review it simply to ensure no MAJOR logic errors exist compared to the user's request.
            
            **CRITICAL INSTRUCTION**: 
            - If the query logic seems correct and reasonable, **DO NOT CHANGE IT**. 
            - Only modify the query if you find a **DEFINITIVE** logical contradiction with the question.
            - Do not over-correct based on the limited data sample shown in the current situation.

            User Question: {query}
            Schema Information: {schema_info}
            Candidate SQL: {previous_code}
            Current situation: {error}
            Error history: {error_history}

            **Decision Protocol**:
            1. **Logic Check**: Does the SQL filter for what was asked? 
            2. **Data Sample Check**: Does the returned sample data seem correct? (Attention: the answer can be **empty**.)
            
            **Output**:
            - If Logic is Sound: Output the **Original SQL** unchanged.
            - If Major Flaw Found: Output the **Corrected SQL**..

            **Output Format**:
            
            Analysis: [Explain why you keep it or change it]

            ```sql
            (The Final SQL Query)
            ``` 
            """

            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "few_shots": self.few_shot_examples,
                "query": self.query, 
                "schema_info": self.schema_info,
                "previous_code": self.response,
                "error": error,
                "error_history": self.error_history,
            })
        
        return extract_code(response)
    

    def reflect(self, error: str):
        history = self.error_history
        if len(history) > 350000:
            history = history[:350000] + "\n...(truncated due to length limit)..."
        
        prompt_template = """
        You are an expert Reviewer. Your goal is to summarize the debugging progress and provide actionable suggestions for the next step.

        **Context**:
        1. **Previous Error History (Summarized)**: 
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

        **Output Format:**(if Current Error is "No error", only output Error History part)
        Error History:
        1. Error: [Summarized Contents]. Solution: [Summarized Contents].
        2. Error: [Summarized Contents]. Solution: [Summarized Contents].
        ...
        Latest blocker:
        Previous fix [Worked/Failed]. Current blocker is [Error]. Suggestion: [Action].
        """
        
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


        return response

    def run(self, query: str, evidence: str, metadata_str: str, schema_info: str, attempt: int, error: str, exe_flag: bool):  
        with get_openai_callback() as cb:
            reference = ""
            if attempt == 1:
                self.query = query   
                self.error_history = ""
                self.last_code = ""
                self.last_error = ""
                self.schema_info = schema_info
                context = ""

                if evidence:
                    context += f"\nExternal Knowledge: {evidence}\n"
                
                # context += "--- knowledge for related terms from wikipedia(only for reference) ---\n"
                # context += self.get_knowledge()         
        
                self.response = self.original(context, metadata_str)

            elif exe_flag is False:
                self.error_history = self.reflect(error)
                self.last_code = self.response
                self.last_error = error
                if "Query executed successfully but returned an empty DataFrame" in error:
                    reference = """                
                    **Double Check for Empty Result**:
                    1. **Case Sensitivity (Critical)**: SQLite text comparison is case-sensitive by default (e.g., 'Google' != 'google'). **FIX**: Use `LOWER(col) = 'val'` or `LIKE 'val'`.
                    2. **Date Format**: Do NOT assume 'YYYY-MM-DD' or 'YYYYMMDD'. Use `STRFTIME` or `SUBSTR` to align formats.
                    3. **Hidden Spaces**: Use `TRIM(col)` if column values might have spaces.
                    4. **NO Defensive Logic**: **NEVER** modify the query to return a string like 'No data found'. If the result is empty, it should remain empty. Just check the filter logic.
                    """
                if "timeout" in error:
                    reference = """
                    **Performance Optimization Required**: 
                    1. **Avoid Subqueries in SELECT**: Move correlated subqueries to `JOIN` or `CTE`.
                    2. **Simplify JOINS**: Do not join tables that aren't needed for the final SELECT or WHERE.
                    3. **String Function**: Avoid complex string manipulation in `WHERE` clauses on large tables if possible.
                    """
                else:
                    reference = ""
                self.response = self.revise(error, reference, exe_flag)
            else:
                self.error_history = self.reflect("No error.")
                self.last_code = self.response
                self.last_error = error
                self.response = self.revise("", reference, exe_flag)

            token_use = f"Total: {cb.total_tokens}, Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens} | Cost: ${cb.total_cost:.4f}\n"
            print(token_use)
        print(self.response)
        return self.response, token_use
        
