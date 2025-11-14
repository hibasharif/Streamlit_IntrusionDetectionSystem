
# Intrusion Detection Streamlit App (Prototype)

This folder contains a Streamlit-based web app scaffold for your Intrusion Detection final-year project.
Features included:
- Upload or train a quick model (LogisticRegression) with dataset upload
- Batch prediction from CSV and download results
- Animated UI elements and theme support (dark / light / blue)
- Confusion matrix sample visualization and classification report
- Saves trained quick-model to `/mnt/data/trained_model.pkl`

To run locally:
1. Create a virtual environment and install requirements: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Place any project dataset in the same folder or upload via the sidebar.
