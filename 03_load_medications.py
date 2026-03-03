import pandas as pd
import os
import time
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_PATH = os.getenv("BASE_PATH")

# 2. Setup LLM (Using the working Llama 3.3 model)
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# 3. Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def standardize_drug(drug_name):
    """
    Uses LLM to extract the generic ingredient from a brand/dosage string.
    Example: "Tylenol 500mg" -> "Acetaminophen"
    """
    system_prompt = """
    You are a pharmacy expert. 
    Extract the GENERIC INGREDIENT name from the following drug string.
    - Ignore dosages (500mg).
    - Ignore form (tablet, IV).
    - If it is a brand name, convert to generic (e.g., Lasix -> Furosemide).
    - Return ONLY the generic name. No extra text.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": drug_name})
        return response.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return drug_name  # Fallback to original name


def add_prescription_to_graph(tx, row, generic_name):
    """
    Links Admission to an Ingredient.
    """
    query = """
    MATCH (a:Admission {hadm_id: $hid})
    MERGE (d:Ingredient {name: $generic})
    MERGE (a)-[:PRESCRIBED {
        drug_raw: $raw, 
        dose: $dose, 
        route: $route,
        start: $start,
        end: $end
    }]->(d)
    """
    tx.run(query,
           hid=str(row['hadm_id']),
           generic=generic_name,
           raw=row['drug'],
           dose=str(row['dose_val_rx']),
           route=str(row['route']),
           start=str(row['starttime']),
           end=str(row['stoptime']))


def main():
    print("1. Loading Prescriptions...")
    df_prescriptions = pd.read_csv(f"{BASE_PATH}/hosp/prescriptions.csv")

    # Filter for non-null drugs
    df_prescriptions = df_prescriptions.dropna(subset=['drug'])

    # Slicing for demo (process first 50 rows only to test)
    # Remove .head(50) when you are ready to run all
    batch_df = df_prescriptions.head(50)

    print(f"2. Processing {len(batch_df)} prescriptions...")

    drug_cache = {}

    with driver.session() as session:
        for index, row in batch_df.iterrows():
            raw_drug = row['drug']

            # Check cache to save API calls
            if raw_drug in drug_cache:
                generic = drug_cache[raw_drug]
            else:
                print(f"   Standardizing: {raw_drug}...")
                generic = standardize_drug(raw_drug)
                drug_cache[raw_drug] = generic
                time.sleep(0.5)  # Rate limit

            session.execute_write(add_prescription_to_graph, row, generic)

    print("Success! Medication graph built.")
    driver.close()


if __name__ == "__main__":
    main()