# Proteomics Analysis Pipeline

A Streamlit application for preprocessing and differential analysis of DDA
proteomics data.

## Main functions

- Upload proteomics data in CSV format
- Missing-value imputation
- Data normalization
- Coefficient-of-variation filtering
- Statistical testing and multiple-testing correction
- PCA and volcano plots
- Gene Ontology enrichment
- Downloadable results and figures

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application normally opens at `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Select the repository and branch.
4. Set the application entry point to `app.py`.
5. Deploy the application.

## Deploy with Google Cloud Run

The included `Dockerfile` can be deployed from the local folder:

```bash
gcloud run deploy proteomics-pipeline \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3600 \
  --session-affinity
```

## Data privacy

Do not commit patient-identifiable, clinical, or experimental datasets to the
repository. The `.gitignore` excludes common tabular-data formats, local
secrets, virtual environments, and generated results.

This application provides research and analytical outputs. It is not a
diagnostic medical device and should not be used as a substitute for
professional clinical judgment.
