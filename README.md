# 🤪 GCN-Based Molecule Property Predictor

A Streamlit web app that predicts key chemical properties — `logP`, `QED`, and `SAS` — directly from SMILES strings using a trained **Graph Convolutional Network (GCN)** on the ZINC250K dataset.

---

## 🌟 Features

* 🧠 Predict logP, qed, and SAS values using deep learning
* 📄 Upload CSV files with SMILES for batch prediction
* 🔬 Visualize molecular structure from SMILES input
* 💻 Built with PyTorch, DGL, dgllife, and RDKit
* 🚀 Deployed via Streamlit + Ngrok (local tunnel)

---

## 📷 Live Demo

👉 **[Try the App Here](https://9304f39325e0.ngrok-free.app)**
*(This link is active while Ngrok is running)*

---

## ⚙️ Tech Stack

* Python 3.11
* [DGL](https://www.dgl.ai/) (Deep Graph Library)
* [dgllife](https://lifesci.dgl.ai/)
* PyTorch
* RDKit
* Streamlit

---

## 📅 How to Run Locally

```bash
git clone https://github.com/your-username/gcn-molecule-property-predictor.git
cd gcn-molecule-property-predictor

# (Optional) Create virtual env
python -m venv gcn-env
gcn-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Example CSV Format

```csv
smiles
CCO
c1ccccc1
CC(C)C(=O)O
```

---

## 🧑‍💼 Author

Made with ❤️ by **Bhaskar146**
IIT Kanpur | GitHub: [@Bhaskar146](https://github.com/Bhaskar146)

---

## 📄 License

This project is for academic use only.
