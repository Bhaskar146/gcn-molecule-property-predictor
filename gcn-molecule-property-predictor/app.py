import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import numpy as np
from dgllife.utils import smiles_to_bigraph
from dgllife.utils.featurizers import atom_type_one_hot, bond_type_one_hot, BaseAtomFeaturizer, BaseBondFeaturizer

# Import your model class (paste it into same file or import properly)
class GCNMultiRegressor(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCNMultiRegressor, self).__init__()
        from dgl.nn.pytorch import GraphConv
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)
        self.readout = torch.nn.Linear(hidden_feats, out_feats)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        g.ndata['h'] = x
        hg = g.ndata['h'].mean(0, keepdim=True)
        return self.readout(hg)

# Load trained model
model = GCNMultiRegressor(in_feats=43, hidden_feats=64, out_feats=3)
model.load_state_dict(torch.load("gcn_multitask_model_1000.pt", map_location=torch.device('cpu')))
model.eval()

# Featurizers
atom_f = BaseAtomFeaturizer({'h': atom_type_one_hot})
bond_f = BaseBondFeaturizer({'e': bond_type_one_hot})

st.title("Molecule Property Predictor (logP, qed, SAS)")
smiles = st.text_input("Enter a SMILES string (e.g., CCO, c1ccccc1):")

if smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))

        g = smiles_to_bigraph(smiles.strip(), add_self_loop=False,
                              node_featurizer=atom_f, edge_featurizer=bond_f)

        with torch.no_grad():
            pred = model(g, g.ndata['h']).numpy().flatten()

        # Layout using Streamlit columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Molecule Preview", use_column_width=True)

        with col2:
            st.subheader("Predicted Properties")
            st.metric("logP", f"{pred[0]:.4f}")
            st.metric("qed", f"{pred[1]:.4f}")
            st.metric("SAS", f"{pred[2]:.4f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
# --- CSV Upload for Batch Prediction from 250k_rndm_zinc_drugs_clean_3.csv ---
st.markdown("---")
st.subheader("📁 Upload ZINC250K Subset for Prediction")

uploaded_file = st.file_uploader("Upload '250k_rndm_zinc_drugs_clean_3.csv' or similar CSV", type="csv")

if uploaded_file:
    import pandas as pd

    try:
        df = pd.read_csv(uploaded_file)

        if 'smiles' not in df.columns:
            st.error("❌ CSV must contain a 'smiles' column.")
        else:
            st.write("✅ Uploaded File Preview:")
            st.dataframe(df.head())

            batch_results = []

            for smi in df['smiles']:
                try:
                    g = smiles_to_bigraph(smi.strip(), add_self_loop=False,
                                          node_featurizer=atom_f, edge_featurizer=bond_f)

                    with torch.no_grad():
                        pred = model(g, g.ndata['h']).numpy().flatten()

                    batch_results.append({
                        'smiles': smi,
                        'pred_logP': pred[0],
                        'pred_qed': pred[1],
                        'pred_SAS': pred[2]
                    })

                except Exception as e:
                    batch_results.append({
                        'smiles': smi,
                        'pred_logP': 'error',
                        'pred_qed': 'error',
                        'pred_SAS': 'error'
                    })

            result_df = pd.DataFrame(batch_results)
            st.success("📊 Batch Prediction Completed")
            st.dataframe(result_df)

            # Optional: Add download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predicted CSV", data=csv, file_name='zinc_predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"❌ Failed to read/process file: {e}")
