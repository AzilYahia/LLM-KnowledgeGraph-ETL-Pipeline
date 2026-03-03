import pandas as pd
import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_PATH = os.getenv("BASE_PATH")

llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


# --- 2. HELPER FUNCTIONS (LLM) ---
def get_llm_response(system_prompt, user_text):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")])
    chain = prompt | llm
    try:
        return chain.invoke({"text": user_text}).content.strip()
    except:
        return "Unknown"


def standardize_category(text):
    return get_llm_response(
        "Classify diagnosis into ONE category: [Cardiovascular, Respiratory, Neurological, Digestive, Infectious, Other]. Return ONLY the category.",
        text
    )


def standardize_drug(text):
    return get_llm_response(
        "Extract GENERIC ingredient name from this drug string. Return ONLY the name.",
        text
    )


def interpret_lab(name, val, unit):
    return get_llm_response(
        "Given test, value, unit, provide ONE-WORD interpretation (e.g. High, Normal, Low).",
        f"{name} {val} {unit}"
    )


# --- 3. DATABASE FUNCTIONS ---
def clear_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def load_patient_skeleton(tx, row):
    tx.run("""
    MERGE (p:Patient {subject_id: $sid}) 
    SET p.gender = $gender, p.age = $age
    """, sid=str(row['subject_id']), gender=row['gender'], age=row['anchor_age'])


def load_admission(tx, row):
    tx.run("""
    MATCH (p:Patient {subject_id: $sid})
    MERGE (a:Admission {hadm_id: $hid})
    MERGE (p)-[:HAS_ADMISSION]->(a)
    """, sid=str(row['subject_id']), hid=str(row['hadm_id']))


def load_diagnosis(tx, hid, code, title, category):
    tx.run("""
    MATCH (a:Admission {hadm_id: $hid})
    MERGE (d:Disease {name: $title}) SET d.category = $cat
    MERGE (a)-[:DIAGNOSED_WITH]->(d)
    """, hid=str(hid), title=title, cat=category)


def load_med(tx, hid, drug_raw, generic):
    tx.run("""
    MATCH (a:Admission {hadm_id: $hid})
    MERGE (m:Ingredient {name: $gen})
    MERGE (a)-[:PRESCRIBED {raw: $raw}]->(m)
    """, hid=str(hid), gen=generic, raw=drug_raw)


def load_lab(tx, hid, name, val, unit, interp):
    tx.run("""
    MATCH (a:Admission {hadm_id: $hid})
    CREATE (r:LabResult {value: $val, unit: $unit, interpretation: $interp})
    MERGE (t:LabTest {name: $name})
    MERGE (a)-[:HAS_LAB_RESULT]->(r)-[:IS_FOR_TEST]->(t)
    """, hid=str(hid), val=str(val), unit=str(unit), interp=interp, name=name)


# --- 4. MAIN PIPELINE ---
def main():
    print("--- STARTING TARGETED PIPELINE ---")

    # A. LOAD CSVs
    print("Reading CSVs...")
    df_pat = pd.read_csv(f"{BASE_PATH}/hosp/patients.csv")
    df_adm = pd.read_csv(f"{BASE_PATH}/hosp/admissions.csv")
    df_diag = pd.read_csv(f"{BASE_PATH}/hosp/diagnoses_icd.csv")
    df_dict = pd.read_csv(f"{BASE_PATH}/hosp/d_icd_diagnoses.csv")
    df_med = pd.read_csv(f"{BASE_PATH}/hosp/prescriptions.csv")
    df_lab = pd.read_csv(f"{BASE_PATH}/hosp/labevents.csv")
    df_lab_items = pd.read_csv(f"{BASE_PATH}/hosp/d_labitems.csv")

    # B. SELECT TARGET PATIENT (Find one with data in ALL tables)
    # We find intersection of subject_ids
    common_ids = set(df_pat.subject_id) & set(df_diag.subject_id) & set(df_med.subject_id) & set(df_lab.subject_id)
    target_id = list(common_ids)[0]  # Pick the first one
    print(f"\n>>> TARGET PATIENT IDENTIFIED: {target_id} <<<")

    # C. FILTER DATA FOR THIS PATIENT
    p_pat = df_pat[df_pat.subject_id == target_id]
    p_adm = df_adm[df_adm.subject_id == target_id]
    p_diag = df_diag[df_diag.subject_id == target_id].merge(df_dict, on='icd_code', how='left')
    p_med = df_med[df_med.subject_id == target_id].dropna(subset=['drug'])

    # Labs: Join with names and filter for interesting ones
    interesting_labs = [50912, 50983, 50809, 51221, 51301]  # Creatinine, Sodium, Glucose, Hct, WBC
    p_lab = df_lab[(df_lab.subject_id == target_id) & (df_lab.itemid.isin(interesting_labs))]
    lab_map = pd.Series(df_lab_items.label.values, index=df_lab_items.itemid).to_dict()

    print(f"Data to Process: {len(p_adm)} Admissions, {len(p_diag)} Diagnoses, {len(p_med)} Meds, {len(p_lab)} Labs")

    # D. EXECUTE GRAPH BUILD
    with driver.session() as session:
        print("\n1. Clearing DB & Loading Skeleton...")
        session.execute_write(clear_db)

        for _, row in p_pat.iterrows(): session.execute_write(load_patient_skeleton, row)
        for _, row in p_adm.iterrows(): session.execute_write(load_admission, row)

        print("2. Processing Diagnoses (LLM)...")
        for _, row in p_diag.head(10).iterrows():  # Limit to 10 for speed
            title = row['long_title'] if pd.notna(row['long_title']) else "Unknown"
            cat = standardize_category(title)
            print(f"   - {title[:30]}... -> {cat}")
            session.execute_write(load_diagnosis, row['hadm_id'], row['icd_code'], title, cat)

        print("3. Processing Medications (LLM)...")
        for _, row in p_med.head(10).iterrows():
            gen = standardize_drug(row['drug'])
            print(f"   - {row['drug'][:30]}... -> {gen}")
            session.execute_write(load_med, row['hadm_id'], row['drug'], gen)

        print("4. Processing Labs (LLM)...")
        for _, row in p_lab.head(15).iterrows():
            name = lab_map.get(row['itemid'], "Unknown")
            interp = interpret_lab(name, row['valuenum'], row['valueuom'])
            print(f"   - {name}: {row['valuenum']} -> {interp}")
            session.execute_write(load_lab, row['hadm_id'], name, row['valuenum'], row['valueuom'], interp)

    print("\n--- PIPELINE COMPLETE. RUNNING SUMMARY ---")
    # Run the summary script immediately
    os.system(f"python 05_patient_summary_rag.py")


if __name__ == "__main__":
    main()