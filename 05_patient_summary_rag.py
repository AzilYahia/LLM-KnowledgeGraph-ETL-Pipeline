import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Secrets
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Setup LLM
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# 3. Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def get_patient_data(tx):
    """
    Retrieves ALL data for the patient, aggregating across all their admissions.
    This prevents the 'missing data' error if labs and meds are in different visits.
    """
    query = """
    MATCH (p:Patient)

    // Optional matches allow us to retrieve the patient even if some data is missing
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission)
    OPTIONAL MATCH (a)-[:DIAGNOSED_WITH]->(d:Disease)
    OPTIONAL MATCH (a)-[pr:PRESCRIBED]->(m:Ingredient)
    OPTIONAL MATCH (a)-[:HAS_LAB_RESULT]->(lr:LabResult)-[:IS_FOR_TEST]->(lt:LabTest)

    // Aggregate everything at the Patient level
    WITH p, 
         collect(DISTINCT d.description + " (" + d.category + ")") AS diagnoses,
         collect(DISTINCT m.name) AS medications,
         collect(DISTINCT lt.name + ": " + lr.interpretation) AS labs

    RETURN p.subject_id AS subject_id, 
           p.gender AS gender, 
           p.age AS age,
           diagnoses[0..15] as diagnoses,   // Limit to 15 to fit in LLM context
           medications[0..15] as medications, 
           labs[0..15] as labs
    LIMIT 1
    """
    result = tx.run(query)
    return [record.data() for record in result]


def generate_clinical_summary(patient_context):
    """
    Generates the Doctor's Note using Graph RAG.
    """
    system_prompt = """
    You are an expert Chief Medical Officer.
    I will provide you with a patient's clinical knowledge graph data.

    Your task: Write a professional "Hospital Discharge Summary" for this patient.

    Structure:
    1. Patient Demographics
    2. Primary Diagnoses (Group by system like 'Infectious', 'Cardio')
    3. Key Medications
    4. Critical Lab Findings (Focus on 'High', 'Low', 'Elevated')
    5. Clinical Assessment (Synthesize the connection between the labs and the diseases).

    Keep it concise and professional.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Patient Data:\n{context}")
    ])

    chain = prompt | llm

    print("\n" + "=" * 40)
    print("      GENERATING AI CLINICAL REPORT      ")
    print("=" * 40 + "\n")

    try:
        response = chain.invoke({"context": str(patient_context)})
        print(response.content)
    except Exception as e:
        print(f"LLM Error: {e}")


def main():
    print("1. Querying Neo4j Knowledge Graph...")
    with driver.session() as session:
        patient_records = session.execute_read(get_patient_data)

    if not patient_records:
        print("Error: No patient found in the database. Please run script 00 first.")
        return

    patient_data = patient_records[0]

    print(f"   -> Found Patient: {patient_data['subject_id']}")
    print(f"   -> Diagnoses: {len(patient_data['diagnoses'])}")
    print(f"   -> Meds: {len(patient_data['medications'])}")
    print(f"   -> Labs: {len(patient_data['labs'])}")

    # Run the RAG
    generate_clinical_summary(patient_data)

    driver.close()


if __name__ == "__main__":
    main()