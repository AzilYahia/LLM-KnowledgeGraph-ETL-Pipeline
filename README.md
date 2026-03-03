# LLM-Assisted Medical Knowledge Graph Pipeline (MIMIC-IV)

## 📌 Project Overview
This project implements an **Intelligent ETL Pipeline** that transforms raw Electronic Health Records (MIMIC-IV) into a **Heterogeneous Knowledge Graph**. It leverages Large Language Models (LLMs) to enrich clinical data (normalizing medications, interpreting labs) and uses **Graph Neural Networks (GNNs)** to generate dense patient embeddings for downstream predictive tasks.

**Thesis:** Sujet 7 – Un pipeline ETL assisté par LLM pour la construction et l’enrichissement d’un graphe de connaissances médical.

## 🚀 Key Features
* **Intelligent Extraction:** Converts CSVs to Graph Nodes (Patient, Admission, Disease, Meds).
* **Semantic Enrichment:** Uses **Llama-3-70b** (via Groq) to classify diseases and standardize drug names.
* **Graph RAG:** Retrieval-Augmented Generation for automated clinical summary writing.
* **Graph Representation Learning:** A **Heterogeneous GraphSAGE** model to create patient "Digital Twins" (Embeddings).

## 🛠️ Tech Stack
* **Database:** Neo4j Community Edition
* **LLM Engine:** Groq API (Llama-3) / LangChain
* **GNN Framework:** PyTorch Geometric (PyG)
* **Language:** Python 3.9+

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/AzilYahia/mimic-graph-etl.git](https://github.com/your-username/mimic-graph-etl.git)
    cd mimic-graph-etl
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    GROQ_API_KEY=gsk_your_groq_key
    BASE_PATH=./mimic-iv-clinical-database-demo-2.2
    ```
## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `00_load_batch_50.py` | **Main ETL Script.** Extracts MIMIC data, enriches with LLM, loads to Neo4j. |
| `05_patient_summary_rag.py` | **Graph RAG.** Generates clinical summaries using the Knowledge Graph. |
| `06_export_to_pyg.py` | **Graph Export.** Converts Neo4j graph to PyTorch Geometric (PyG) format. |
| `07_gnn_model.py` | **GNN Model.** Trains a Heterogeneous GraphSAGE to generate patient embeddings. |
| `app_gui.py` | **User Interface.** A GUI to run the pipeline without code. |
| `requirements.txt` | Python dependencies. |


## 🏃 Usage Guide

**Step 1: Build the Graph (ETL)**
Extracts data, runs LLM enrichment, and populates Neo4j.
```bash
python 00_load_batch_50.py
```
**Step 2: Generate Clinical Summaries (RAG)**
Tests the graph by asking the LLM to summarize a patient's history.
```bash
python 05_patient_summary_rag.py
```

**Step 3: Export to PyTorch**
Converts the Neo4j graph into a PyG HeteroData object.
```bash
python 06_export_to_pyg.py
```


**Step 4: Train GNN / Generate Embeddings**
Runs the GraphSAGE model to produce patient vectors.
```bash
python 07_gnn_model.py
```



















