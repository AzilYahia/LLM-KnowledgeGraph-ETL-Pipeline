# LLM-Assisted Medical Knowledge Graph Pipeline (MIMIC-IV)

## 📌 Project Overview
This project implements an **Intelligent ETL Pipeline** that transforms raw Electronic Health Records (MIMIC-IV) into a **Heterogeneous Knowledge Graph**. It leverages Large Language Models (LLMs) to enrich clinical data and uses **Graph Neural Networks (GNNs)** to generate dense patient embeddings.

**Thesis:** Un pipeline ETL assisté par LLM pour la construction et l’enrichissement d’un graphe de connaissances médical.



## 🚀 Key Features
* **Intelligent Extraction:** Converts MIMIC-IV CSVs to Graph Nodes (Patient, Admission, Disease, Meds).
* **Semantic Enrichment:** Uses **Llama-3-70b** to classify diseases and standardize drug names.
* **Graph Representation Learning:** Implements **Heterogeneous GraphSAGE** to create patient "Digital Twins".
* **Phenotyping:** Automated similarity analysis to identify clinically similar patients (Patient Twins).
* **Graph RAG:** Automated clinical summary generation using retrieved graph context.

## 🛠️ Tech Stack
* **Database:** Neo4j Community Edition
* **LLM Engine:** Groq API (Llama-3) / LangChain
* **GNN Framework:** PyTorch Geometric (PyG)
* **Hardware Acceleration:** Supports NVIDIA CUDA (RTX 3060 Ti optimized)

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `00_load_batch_50.py` | **Main ETL Script.** Loads 50 patients with LLM enrichment. |
| `05_patient_summary_rag.py` | **Graph RAG.** Generates patient summaries from Neo4j. |
| `06_export_to_pyg.py` | **Graph Export.** Converts Neo4j to PyTorch Geometric (`.pt`). |
| `07_gnn_model.py` | **GNN Model.** Generates 32-dimensional patient vectors. |
| `08_find_similar_patients.py`| **Analysis.** Calculates Cosine Similarity between patient vectors. |
| `check_graph_stats.py` | **DB Stats.** Displays total node and edge counts. |
| `app_gui.py` | **User Interface.** Modern GUI for pipeline management. |

## ⚙️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/AzilYahia/LLM-KnowledgeGraph-ETL-Pipeline.git](https://github.com/AzilYahia/LLM-KnowledgeGraph-ETL-Pipeline.git)
    cd LLM-KnowledgeGraph-ETL-Pipeline
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables** (`.env`):
    ```ini
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    GROQ_API_KEY=gsk_your_key
    BASE_PATH=./path_to_mimic_data
    ```

## 🏃 Execution Workflow

1. **Build Graph:** `python 00_load_batch_50.py` (Loads ~300 nodes into Neo4j).
2. **Verify Stats:** `python check_graph_stats.py` (Expected: 50 Patients).
3. **Prepare GNN:** `python 06_export_to_pyg.py` (Creates `mimic_gnn_data.pt`).
4. **Generate Embeddings:** `python 07_gnn_model.py`.
5. **Analyze Twins:** `python 08_find_similar_patients.py` (Finds similar patient phenotypes).

---
*Developed as part of a Master's Thesis in AI & Medical Informatics.*