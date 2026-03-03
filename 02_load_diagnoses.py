import pandas as pd
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import time
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_PATH = os.getenv("BASE_PATH")


# --- SETUP LLM ---
# We use Mixtral-8x7b (very smart, very fast)
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# --- NEO4J CONNECTION ---
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def get_disease_category(diagnosis_text):
    """
    Uses LLM to categorize a disease name into a standard medical body system.
    """
    system_prompt = """
    You are a medical coding assistant. 
    Classify the following diagnosis into ONE of these categories:
    [Cardiovascular, Respiratory, Neurological, Digestive, Endocrine, Renal, Infectious, Musculoskeletal, Trauma, Other].

    Return ONLY the category name. No other text.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": diagnosis_text})
        return response.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Unknown"


def add_diagnosis_to_graph(tx, row, category):
    """
    1. Creates a Disease node (if not exists).
    2. Links Admission to Disease.
    """
    query = """
    MATCH (a:Admission {hadm_id: $hid})
    MERGE (d:Disease {code: $code})
    ON CREATE SET d.description = $desc, d.category = $cat
    MERGE (a)-[:DIAGNOSED_WITH {seq_num: $seq}]->(d)
    """
    tx.run(query,
           hid=str(row['hadm_id']),
           code=str(row['icd_code']),
           desc=row['long_title'],
           cat=category,
           seq=int(row['seq_num']))


def main():
    print("1. Loading Diagnosis Data...")
    # Load diagnoses (codes linked to admissions)
    df_diag = pd.read_csv(f"{BASE_PATH}/hosp/diagnoses_icd.csv")

    # Load dictionary (codes linked to text descriptions)
    df_dict = pd.read_csv(f"{BASE_PATH}/hosp/d_icd_diagnoses.csv")

    # Join them to get the text description
    # Note: MIMIC splits codes into ICD9 and ICD10. For the demo, we merge on code.
    # We strip whitespace just in case.
    df_diag['icd_code'] = df_diag['icd_code'].str.strip()
    df_dict['icd_code'] = df_dict['icd_code'].str.strip()

    merged_df = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='left')

    # Take a small sample for testing (e.g., first 50 rows)
    # If you want to run all, remove .head(50)
    batch_df = merged_df.head(50)

    print(f"2. Processing {len(batch_df)} diagnoses with LLM...")

    # Cache to avoid re-asking LLM for the same disease
    category_cache = {}

    with driver.session() as session:
        for index, row in batch_df.iterrows():
            disease_name = row['long_title']
            code = row['icd_code']

            # Skip if description is missing
            if pd.isna(disease_name):
                disease_name = "Unknown Diagnosis"

            # Check cache first
            if disease_name in category_cache:
                category = category_cache[disease_name]
            else:
                print(f"   Asking LLM about: {disease_name}...")
                category = get_disease_category(disease_name)
                category_cache[disease_name] = category
                # Rate limit to be nice to the API
                time.sleep(0.5)

            # Write to Neo4j
            session.execute_write(add_diagnosis_to_graph, row, category)

    print("Success! Diagnoses graph enriched.")
    driver.close()


if __name__ == "__main__":
    main()