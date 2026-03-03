import os
import torch
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from dotenv import load_dotenv
from tqdm import tqdm  # For progress bars

# 1. Setup
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# Load a small, fast BERT model for embeddings
print("Loading Embedding Model (MiniLM)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')


def get_nodes(tx, label, property_key):
    """Fetches all nodes of a specific label and their text property."""
    query = f"MATCH (n:{label}) RETURN id(n) as neo_id, n.{property_key} as text"
    result = tx.run(query)
    return pd.DataFrame([r.data() for r in result])


def get_edges(tx, src_label, rel_type, dst_label):
    """Fetches all edges between two node types."""
    query = f"""
    MATCH (a:{src_label})-[r:{rel_type}]->(b:{dst_label})
    RETURN id(a) as src_id, id(b) as dst_id
    """
    result = tx.run(query)
    return pd.DataFrame([r.data() for r in result])


def main():
    data = HeteroData()

    with driver.session() as session:
        print("--- 1. Processing Nodes & Features ---")

        # Define the node types we want in our GNN
        # Format: (Neo4j Label, Property to Embed)
        node_configs = [
            ("Patient", "gender"),  # Embedding "M"/"F"
            ("Disease", "name"),  # Embedding "Heart Failure"
            ("Ingredient", "name"),  # Embedding "Aspirin"
            ("LabTest", "name")  # Embedding "Glucose"
        ]

        # We need to map Neo4j IDs (integer) to PyG consecutive IDs (0, 1, 2...)
        id_mapping = {}

        for label, prop in node_configs:
            print(f"   Fetching {label}...")
            df = session.execute_read(get_nodes, label, prop)

            if df.empty:
                print(f"   Warning: No nodes found for {label}")
                continue

            # Fill missing text
            df['text'] = df['text'].fillna("Unknown")

            # Create mapping: Neo4j ID -> PyG Index
            # Example: {105: 0, 108: 1, ...}
            mapping = {nid: i for i, nid in enumerate(df['neo_id'])}
            id_mapping[label] = mapping

            # Generate Embeddings (Text -> Vector)
            # This turns "Heart Failure" into a vector of 384 numbers
            embeddings = encoder.encode(df['text'].tolist(), show_progress_bar=True)

            # Add to PyG Data Object
            # data['Patient'].x stores the features
            data[label].x = torch.tensor(embeddings, dtype=torch.float)
            print(f"   -> Added {len(df)} {label} nodes.")

        print("\n--- 2. Processing Edges (Topology) ---")

        # Define the edges we want
        edge_configs = [
            ("Patient", "HAS_ADMISSION", "Admission"),  # Skip Admission node for simplicity?
            # Actually, let's simplify: Patient -> Disease, Patient -> Med
            # But in our graph it is Patient -> Admission -> Disease
            # For GNN, let's "Collapse" the Admission node to make it easier
            # Query: MATCH (p:Patient)-[:HAS_ADMISSION]->(:Admission)-[:DIAGNOSED_WITH]->(d:Disease)
        ]

        # Let's run a custom query to get Patient->Disease edges directly
        # This is "Graph Projection"

        # 2a. Patient -> Disease
        print("   Linking Patients -> Diseases...")
        q_diag = """
        MATCH (p:Patient)-[:HAS_ADMISSION]->(:Admission)-[:DIAGNOSED_WITH]->(d:Disease)
        RETURN id(p) as src, id(d) as dst
        """
        df_diag = pd.DataFrame([r.data() for r in session.run(q_diag)])
        if not df_diag.empty:
            src = [id_mapping['Patient'][i] for i in df_diag['src']]
            dst = [id_mapping['Disease'][i] for i in df_diag['dst']]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['Patient', 'has_disease', 'Disease'].edge_index = edge_index

        # 2b. Patient -> Ingredient
        print("   Linking Patients -> Medications...")
        q_med = """
        MATCH (p:Patient)-[:HAS_ADMISSION]->(:Admission)-[:PRESCRIBED]->(m:Ingredient)
        RETURN id(p) as src, id(m) as dst
        """
        df_med = pd.DataFrame([r.data() for r in session.run(q_med)])
        if not df_med.empty:
            src = [id_mapping['Patient'][i] for i in df_med['src']]
            dst = [id_mapping['Ingredient'][i] for i in df_med['dst']]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            data['Patient', 'took_drug', 'Ingredient'].edge_index = edge_index

    print("\n--- 3. Saving ---")
    print(data)
    torch.save(data, "mimic_gnn_data.pt")
    print("Saved to mimic_gnn_data.pt")
    driver.close()


if __name__ == "__main__":
    main()