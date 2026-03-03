import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_geometric.data import HeteroData


def main():
    print("--- 1. Loading Graph Data ---")
    try:
        # Load safely
        data = torch.load("mimic_gnn_data.pt", weights_only=False)
        print("Success: Graph loaded.")
    except FileNotFoundError:
        print("Error: mimic_gnn_data.pt not found.")
        return

    # --- 2. DATA FIX: Reverse Edges ---
    # We need messages to flow TO the Patient, so we flip the edge direction.
    print("--- 2. Creating Reverse Edges for Message Passing ---")

    # Flip Disease -> Patient
    # .flip(0) swaps source and destination rows
    data['Disease', 'rev_has_disease', 'Patient'].edge_index = \
        data['Patient', 'has_disease', 'Disease'].edge_index.flip(0)

    # Flip Ingredient -> Patient
    data['Ingredient', 'rev_took_drug', 'Patient'].edge_index = \
        data['Patient', 'took_drug', 'Ingredient'].edge_index.flip(0)

    # (Optional) Delete the original forward edges so the model isn't confused
    del data['Patient', 'has_disease', 'Disease']
    del data['Patient', 'took_drug', 'Ingredient']

    print("Edges reversed. Messages will now flow: Disease -> Patient")

    # --- 3. Define the GNN Architecture ---
    class MedGNN(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()

            # HeteroConv: We define convolution for the REVERSE edge types
            self.conv1 = HeteroConv({
                ('Disease', 'rev_has_disease', 'Patient'): SAGEConv((-1, -1), hidden_channels),
                ('Ingredient', 'rev_took_drug', 'Patient'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')  # Sum aggregates all info (Diseases + Meds)

            # Post-Processing Linear Layer
            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            # 1. Message Passing
            x_dict = self.conv1(x_dict, edge_index_dict)

            # 2. Activation Function (ReLU)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

            # 3. Final Linear Layer
            x_dict = {key: self.lin(x) for key, x in x_dict.items()}

            return x_dict

    # --- 4. Initialize Model ---
    model = MedGNN(hidden_channels=64, out_channels=32)

    print("\n--- 5. Running Forward Pass ---")

    with torch.no_grad():
        output_dict = model(data.x_dict, data.edge_index_dict)

    # --- 6. Inspect Results ---
    if 'Patient' in output_dict:
        patient_embedding = output_dict['Patient']
        print("\n>>> FINAL RESULT: Patient Representation Vector <<<")
        print(f"Shape: {patient_embedding.shape} (1 Patient, 32 Dimensions)")
        print(patient_embedding)

        print("\n--- CONCLUSION ---")
        print("Success! This vector is the mathematical summary of the patient's health.")
    else:
        print("Error: The model did not produce a Patient embedding.")


if __name__ == "__main__":
    main()