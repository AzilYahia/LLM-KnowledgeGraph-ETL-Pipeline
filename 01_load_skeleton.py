import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
BASE_PATH = os.getenv("BASE_PATH")


# --- CONNECT TO NEO4J ---
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def clear_database(tx):
    """Wipes the database clean. useful for testing."""
    tx.run("MATCH (n) DETACH DELETE n")


def create_patient(tx, row):
    """Creates a Patient node."""
    query = """
    MERGE (p:Patient {subject_id: $sid})
    SET p.gender = $gender,
        p.anchor_age = $age,
        p.anchor_year = $year
    """
    tx.run(query, sid=str(row['subject_id']), gender=row['gender'],
           age=int(row['anchor_age']), year=int(row['anchor_year']))


def create_admission(tx, row):
    """Creates an Admission node and links it to the Patient."""
    query = """
    MATCH (p:Patient {subject_id: $sid})
    MERGE (a:Admission {hadm_id: $hid})
    SET a.admittime = $admittime,
        a.dischtime = $dischtime,
        a.admission_type = $type
    MERGE (p)-[:HAS_ADMISSION]->(a)
    """
    tx.run(query,
           sid=str(row['subject_id']),
           hid=str(row['hadm_id']),
           admittime=row['admittime'],
           dischtime=row['dischtime'],
           type=row['admission_type'])


def main():
    # 1. Load Data
    print("Loading CSVs...")
    df_patients = pd.read_csv(f"{BASE_PATH}/hosp/patients.csv")
    df_admissions = pd.read_csv(f"{BASE_PATH}/hosp/admissions.csv")

    with driver.session() as session:
        # 2. Reset DB (Optional - good for dev)
        print("Clearing Database...")
        session.execute_write(clear_database)

        # 3. Create Patients
        print(f"Creating {len(df_patients)} Patient nodes...")
        for _, row in df_patients.iterrows():
            session.execute_write(create_patient, row)

        # 4. Create Admissions
        print(f"Creating {len(df_admissions)} Admission nodes...")
        for _, row in df_admissions.iterrows():
            session.execute_write(create_admission, row)

    print("Success! Skeleton Graph built.")
    driver.close()


if __name__ == "__main__":
    main()