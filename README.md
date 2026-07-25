# PharmGAT-v2: Multi-View Graph Attention Network for Clinical Polypharmacy Interaction Profiling

**Model Name**: PharmGAT-v2  
**Technical Version**: PharmGAT-v2-DDI  
**Architecture**: Multi-View Graph Attention Network v2 (Multi-head, hierarchical layers)  
**Mission**: Safeguarding patients by predicting adverse drug-drug interactions (DDIs) and interaction mechanisms.

## Overview
Existing DDI prediction systems rely on manually curated databases covering limited drugs, requiring manual mapping of new drugs and often memorizing interactions instead of learning molecular patterns. **PharmGAT-v2** is a graph neural network architecture that learns molecular-level interaction patterns from fused chemical structures.

The model predicts drug-pair interactions with probability scores and classifies interaction types (potential adverse drug reactions) across 86 mechanisms. It is integrated with a Streamlit application featuring manual drug entry, automated OCR-powered prescription scanning, and patient risk profiling.

### Key Features
- **State-of-the-Art Metrics**: `88.5%` test accuracy, `89.5%` F1-score, and `97.5%` recall on DDI pairs, with `97.0%` accuracy on mechanism classification.
- **Prescription Scanner**: OCR-powered image scanning with fuzzy drug name matching.
- **Risk Profiling**: Patient-specific cardiovascular risk stratification validated on 70,000 patient records.
- **Real-time Inference**: <100ms prediction latency, deployable on consumer hardware.
- **Multi-drug Analysis**: Checks all pairwise combinations for up to 4 concurrent medications.

## Dataset Usage
* **Drug Interaction Data (Training)**: 191,808 interaction pairs and 1,706 molecular structures (`dataset/drugdata/`).
* **CVD Patient Data (Validation)**: 70,000 patient records (`dataset/cardio_base.csv`) used for risk profiling validation.

## Quick Start

### Installation
```bash
# Pin Python 3.12 version
uv python pin 3.12

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Run Model Training
Run the Jupyter notebook `PharmGAT_v2_model_train_local.ipynb` in your editor or via:
```bash
uv run jupyter notebook
```

### Run the Streamlit App
```bash
uv run streamlit run streamlit_app.py
```
