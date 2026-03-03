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

# 2. Setup LLM
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# 3. Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# --- CONFIG: INTERESTING LABS ---
# We focus on these IDs to keep the demo clean:
# 50912: Creatinine (Kidney)
# 50983: Sodium (Electrolytes)
# 50809: Glucose (Sugar)
# 51221: Hematocrit (Blood)
# 51301: White Blood Cells (Infection)
INTERESTING_ITEMIDS = [50912, 50983, 50809, 51221, 51301]


def interpret_lab_result(lab_name, value, unit):
    """
    Uses LLM to give a medical interpretation of the value.
    Example: Glucose 180 mg/dL -> "Hyperglycemia"
    """
    system_prompt = """
    You are a clinical pathologist.
    Given a lab test, value, and unit, provide a ONE-WORD medical interpretation.
    Examples: 
    - Glucose 180 mg/dL -> Hyperglycemia
    - WBC 15.0 K/uL -> Leukocytosis
    - Sodium 140 mEq/L -> Normal
    - Creatinine 2.5 mg/dL -> Elevated

    Return ONLY the one-word medical term.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Test: {name}, Value: {val}, Unit: {unit}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"name": lab_name, "val": value, "unit": unit})
        return response.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Unknown"


def add_lab_to_graph(tx, row, lab_name, interpretation):
    """
    Links Admission -> LabResult -> LabTest
    """
    query = """
    MATCH (a:Admission {hadm_id: $hid})
    MERGE (t:LabTest {itemid: $itemid})
    ON CREATE SET t.name = $name

    CREATE (r:LabResult {
        value: $val, 
        unit: $unit, 
        time: $time,
        interpretation: $interp
    })

    MERGE (a)-[:HAS_LAB_RESULT]->(r)
    MERGE (r)-[:IS_FOR_TEST]->(t)
    """
    tx.run(query,
           hid=str(row['hadm_id']),
           itemid=str(row['itemid']),
           name=lab_name,
           val=str(row['valuenum']),
           unit=str(row['valueuom']),
           time=str(row['charttime']),
           interp=interpretation)


def main():
    print("1. Loading Lab Definitions...")
    df_items = pd.read_csv(f"{BASE_PATH}/hosp/d_labitems.csv")
    # Create a dictionary for quick name lookups: {50912: 'Creatinine', ...}
    item_map = pd.Series(df_items.label.values, index=df_items.itemid).to_dict()

    print("2. Loading Lab Events (Filtered)...")
    df_labs = pd.read_csv(f"{BASE_PATH}/hosp/labevents.csv")

    # Filter only for the interesting labs
    df_filtered = df_labs[df_labs['itemid'].isin(INTERESTING_ITEMIDS)].copy()

    # Remove rows with no numeric value
    df_filtered = df_filtered.dropna(subset=['valuenum'])

    # Slice for demo speed (process top 30)
    batch_df = df_filtered.head(30)

    print(f"3. Processing {len(batch_df)} lab results...")

    with driver.session() as session:
        for index, row in batch_df.iterrows():
            item_id = row['itemid']
            lab_name = item_map.get(item_id, "Unknown Lab")
            value = row['valuenum']
            unit = row['valueuom']

            print(f"   Interpreting: {lab_name} ({value} {unit})...")

            # Ask LLM for interpretation
            interpretation = interpret_lab_result(lab_name, value, unit)
            print(f"   -> {interpretation}")

            # Rate limit
            time.sleep(0.5)

            session.execute_write(add_lab_to_graph, row, lab_name, interpretation)

    print("Success! Lab layer added.")
    driver.close()


if __name__ == "__main__":
    main()