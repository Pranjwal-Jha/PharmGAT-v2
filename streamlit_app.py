import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import requests
import re
import time
import json
import itertools
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from safetensors.torch import load_file
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from difflib import get_close_matches

# 1 CONFIGURATION & PATHS
MODEL_WEIGHTS = 'best/PharmGAT_v2_best_weight.safetensors'
MODEL_CONFIG = 'best/PharmGAT_v2_best_metadata.json'
GRAPH_DATA = 'best/PharmGAT_v2_best_graph_data.pt'
SMILES_PATH = 'dataset/drugdata/drug_smiles.csv'
NAMES_PATH = 'dataset/drugdata/drug_names.csv'
CARDIO_BASE_DATA = 'dataset/cardio_base.csv'
DDI_TYPE_MAP = 'dataset/drugdata/ddi_type_mapping.json'

# DEFAULT_THRESHOLD training notebook function
DEFAULT_THRESHOLD = 0.45


st.set_page_config(
    page_title="PharmGAT-v2: Multi-View Drug Safety Monitor",
    page_icon="💊",
    layout="centered"
)

# Initialize session state for personalized risk profiling
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = "MEDIUM"
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0
if 'risk_factors' not in st.session_state:
    st.session_state.risk_factors = []
if 'enable_risk_profiling' not in st.session_state:
    st.session_state.enable_risk_profiling = False


# 2 MODEL ARCHITECTURE
class GATv2NN(nn.Module):
    """
    EXACT architecture from training notebook - DO NOT MODIFY
    """
    def __init__(self, dims, hidden_dim, n_heads, n_types, dropout=0.3):
        super().__init__()
        
        # View Projectors with BatchNorm
        self.proj_v1 = nn.Sequential(
            nn.Linear(dims['v1'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.proj_v2 = nn.Sequential(
            nn.Linear(dims['v2'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.proj_v3 = nn.Sequential(
            nn.Linear(dims['v3'], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Feature Attention
        self.feat_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # GATv2 Layers with residual connections (EXACT names from training)
        self.gat1 = GATv2Conv(
            hidden_dim, hidden_dim // n_heads, 
            heads=n_heads, concat=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(
            hidden_dim, hidden_dim, 
            heads=1, concat=False, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gat3 = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=1, concat=False, dropout=dropout
        ) 
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Edge Classifier with deeper network
        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        self.head_bin = nn.Linear(hidden_dim // 2, 2)
        self.head_type = nn.Linear(hidden_dim // 2, n_types)
        
        self.dropout = dropout

    def get_node_embeddings(self, x_v1, x_v2, x_v3, edge_index):
        """
        Full-graph forward with residual connections and normalization.
        """
        # Project each view
        h1 = self.proj_v1(x_v1)
        h2 = self.proj_v2(x_v2)
        h3 = self.proj_v3(x_v3)
        
        # Multi-view attention fusion
        stack = torch.stack([h1, h2, h3], dim=1)
        scores = F.softmax(self.feat_attention(stack), dim=1)
        h_fused = torch.sum(stack * scores, dim=1)
        
        # GAT Layer 1 with residual
        h_gat1 = self.gat1(h_fused, edge_index)
        h_gat1 = self.norm1(h_gat1 + h_fused)
        h_gat1 = F.elu(h_gat1)
        
        # GAT Layer 2 with residual
        h_gat2 = self.gat2(h_gat1, edge_index)
        node_emb = self.norm2(h_gat2 + h_fused)

        # GAT Layer 3 with residual
        h_gat3 = self.gat3(node_emb, edge_index)
        node_emb = self.norm3(h_gat3 + h_fused)
        
        return node_emb

    def forward_edges_from_emb(self, node_emb, edge_label_index):
        src, dst = edge_label_index[0], edge_label_index[1]
        edge_feat = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        shared = self.edge_encoder(edge_feat)
        return self.head_bin(shared), self.head_type(shared)

    def forward(self, x_v1, x_v2, x_v3, edge_index, edge_label_index):
        """Full forward pass (for inference)."""
        node_emb = self.get_node_embeddings(x_v1, x_v2, x_v3, edge_index)
        return self.forward_edges_from_emb(node_emb, edge_label_index)


# 4 ROBUST RESOURCE LOADING
@st.cache_resource(show_spinner=False)
def load_system():
    """Load model, metadata, graph data, and drug information."""
    # Load Metadata
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config JSON not found at {MODEL_CONFIG}")

    with open(MODEL_CONFIG, 'r') as f:
        metadata_list = json.load(f)
    metadata = metadata_list[0] if isinstance(metadata_list, list) else metadata_list
    config = metadata.get('training_configuration', metadata.get('training_configuration', {}))
    if not config:
        raise ValueError("Could not find training configuration in metadata JSON")
    print(f"✓ Loaded config: hidden_dim={config['hidden_dim']}, n_heads={config['n_heads']}, n_gat_layers={config['n_gat_layers']}")

    # Load Graph Data
    if not os.path.exists(GRAPH_DATA):
        raise FileNotFoundError(f"Graph data not found at {GRAPH_DATA}")
    try:
        import torch.serialization
        from sklearn.preprocessing import LabelEncoder

        # Method 1: Add to safe globals
        torch.serialization.add_safe_globals([LabelEncoder])
        graph_data = torch.load(GRAPH_DATA, map_location='cpu', weights_only=False)
    except Exception as e:
        # Method 2: Use context manager (fallback)
        print(f"⚠︎ Trying alternative loading method...")
        with torch.serialization.safe_globals([LabelEncoder]):
            graph_data = torch.load(GRAPH_DATA, map_location='cpu', weights_only=False)
    drug_map = graph_data['drug_map']

    print(f"✓ Loaded graph data with {len(drug_map)} drugs")

    # Load SMILES
    if not os.path.exists(SMILES_PATH):
        raise FileNotFoundError(f"SMILES file not found at {SMILES_PATH}")

    smiles_df = pd.read_csv(SMILES_PATH)

    # Load drug names
    name_dict = {}
    if os.path.exists(NAMES_PATH):
        names_df = pd.read_csv(NAMES_PATH)
        name_dict = dict(zip(names_df['drug_id'], names_df['drug_name']))
        print(f"✓ Loaded {len(name_dict)} drug names")
    else:
        name_dict = {drug_id: drug_id for drug_id in drug_map.keys()}
        print("⚠︎  Drug names file not found, using IDs as names")

    # Initialize Model
    model = GATv2NN(
        dims={
            'v1': graph_data['x_v1'].shape[1],
            'v2': graph_data['x_v2'].shape[1],
            'v3': graph_data['x_v3'].shape[1],
        },
        hidden_dim=config['hidden_dim'],
        n_heads=config['n_heads'],
        n_types=graph_data['n_types'],
        dropout=config['dropout']
    )

    # Load weights
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS}")

    state_dict = load_file(MODEL_WEIGHTS)
    load_result = model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Model weight mismatch!\n"
            f"Missing: {load_result.missing_keys}\n"
            f"Unexpected: {load_result.unexpected_keys}"
        )

    model.eval()
    print(f"✓ Model weights loaded perfectly")

    return model, drug_map, smiles_df, name_dict, config, graph_data


# 5 DRUG INFORMATION LOOKUP (PubChem API)
HEADERS = {"User-Agent": "PharmGAT-v2/2.0 (Research & Clinical Safety Application)"}

# Ordered by clinical relevance for a general audience
CLINICAL_SECTIONS = [
    "Drug Indication",
    "Therapeutic Uses",
    "Pharmacology",
    "Mechanism of Action",
]

def _extract_section(sections: list, targets: list) -> str | None:
    """
    Recursively traverse PubChem's section tree and return the first
    StringWithMarkup text that matches any heading in `targets`.
    """
    for section in sections:
        if section.get("TOCHeading") in targets:
            for info in section.get("Information", []):
                swm = info.get("Value", {}).get("StringWithMarkup")
                if swm:
                    return swm[0].get("String", "").strip()
        if "Section" in section:
            result = _extract_section(section["Section"], targets)
            if result:
                return result
    return None

@st.cache_data(ttl=86400)
def get_drug_description(drug_name: str, sentences: int = 2) -> str:
    """
    Strategy:
      1. Resolve drug name → CID via PubChem PUG REST.
      2. Fetch the full PUG View record.
      3. Walk CLINICAL_SECTIONS in priority order; return first hit.
    Args: drug_name: Generic or brand drug name and sentences: Number of sentences to return
    Returns: A concise clinical description string, or a fallback message.
    """
    try:
        name = drug_name.strip().lower()

        cid_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON",
            headers=HEADERS, timeout=5,
        )
        if cid_resp.status_code != 200:
            return "Description not available."

        cid = cid_resp.json()["IdentifierList"]["CID"][0]

        rec_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON",
            headers=HEADERS, timeout=10,
        )
        if rec_resp.status_code != 200:
            return "Description not available."

        all_sections = rec_resp.json().get("Record", {}).get("Section", [])

        # Walk sections in clinical priority order
        for target in CLINICAL_SECTIONS:
            text = _extract_section(all_sections, [target])
            if text:
                parts = re.split(r'(?<=[.!?])\s+', text)
                return " ".join(parts[:sentences]).strip()

        return "Description not available."

    except Exception:
        return "Description not available."


# Sidebar calculation
def calculate_patient_risk_score(age, bp_systolic, bp_diastolic, cholesterol, smoking, activity):
    """
    Calculate patient cardiovascular risk score using clinical guidelines.
    Risk categories are validated against hackathon-provided CVD patient dataset patterns.
    Returns: risk_level (LOW/MEDIUM/HIGH), risk_score (0-100), risk_factors list
    """
    risk_score = 0
    risk_factors = []

    # Age Risk (0-25 points)
    if age >= 65:
        risk_score += 25
        risk_factors.append("Advanced age (≥65)")
    elif age >= 55:
        risk_score += 15
        risk_factors.append("Elevated age (55-64)")
    elif age >= 45:
        risk_score += 8

    # Blood Pressure Risk (0-30 points)
    if bp_systolic >= 160 or bp_diastolic >= 100:
        risk_score += 30
        risk_factors.append("Stage 2 Hypertension (BP ≥160/100)")
    elif bp_systolic >= 140 or bp_diastolic >= 90:
        risk_score += 20
        risk_factors.append("Stage 1 Hypertension (BP ≥140/90)")
    elif bp_systolic >= 130 or bp_diastolic >= 85:
        risk_score += 10
        risk_factors.append("Elevated BP (130-139/85-89)")

    # Cholesterol Risk (0-20 points)
    if cholesterol == 3:  # High
        risk_score += 20
        risk_factors.append("High cholesterol")
    elif cholesterol == 2:  # Above normal
        risk_score += 10
        risk_factors.append("Elevated cholesterol")

    # Smoking Risk (0-15 points)
    if smoking == 1:
        risk_score += 15
        risk_factors.append("Active smoker")

    # Physical Inactivity (0-10 points)
    if activity == 0:
        risk_score += 10
        risk_factors.append("Physically inactive")

    # Determine risk level
    if risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_level, risk_score, risk_factors

@st.cache_data
def load_cvd_population():
    """Load and preprocess the hackathon CVD dataset."""
    df = pd.read_csv(CARDIO_BASE_DATA, sep=';')
    df['age_years'] = (df['age'] / 365).astype(int)
    return df

def get_population_context(df, age, bp_systolic, cholesterol, smoking, activity):
    """
    Find what % of similar patients in the dataset have CVD.
    Returns a dict with population statistics for display.
    """
    # Build a similarity filter — patients within ±10 years, same cholesterol tier
    mask = (
        (df['age_years'].between(age - 10, age + 10)) &
        (df['cholesterol'] == cholesterol) &
        (df['smoke'] == smoking) &
        (df['active'] == activity)
    )
    cohort = df[mask]

    if len(cohort) < 10:  # fallback to just age band if cohort too small
        cohort = df[df['age_years'].between(age - 10, age + 10)]

    if len(cohort) == 0:
        return None

    cvd_rate = cohort['cardio'].mean() * 100
    cohort_size = len(cohort)
    avg_systolic = cohort['ap_hi'].mean()

    return {
        'cvd_rate': cvd_rate,
        'cohort_size': cohort_size,
        'avg_systolic': avg_systolic,
        'total_dataset': len(df)
    }


def get_risk_adjusted_threshold(risk_level):
    """
    Adjust interaction detection threshold based on patient risk.
    Higher risk patients get more conservative thresholds (detect more interactions).
    """
    thresholds = {
        "HIGH": 0.44,    # More conservative - flag more potential interactions
        "MEDIUM": 0.45,  # Standard threshold (matches training)
        "LOW": 0.46      # Keep consistent with training threshold
    }
    return thresholds.get(risk_level, DEFAULT_THRESHOLD)


# 7 PRESCRIPTION IMAGE SCANNER (OCR)
@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader (cached to avoid reloading)."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)  # CPU mode for compatibility
        return reader
    except ImportError:
        st.error("EasyOCR not installed. Run: pip install easyocr")
        return None
    except Exception as e:
        st.warning(f"OCR initialization failed: {e}")
        return None


def extract_text_from_image(image, reader):
    """Extract text from prescription image using OCR."""
    if reader is None:
        return []

    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Perform OCR
        results = reader.readtext(img_array)

        # Extract text (results format: [(bbox, text, confidence), ...])
        extracted_texts = [text for (bbox, text, conf) in results if conf > 0.3]

        return extracted_texts
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return []


def match_drugs_from_text(extracted_texts, drug_names_dict, drug_map):
    """
    Match extracted text to drug names in database using fuzzy matching.
    Returns list of matched drug IDs.
    """
    matched_drugs = []

    # Create searchable list of drug names (lowercase) - handle non-string names
    drug_names_lower = {}
    for drug_id, name in drug_names_dict.items():
        # Convert to string if not already (handles float/int drug names)
        name_str = str(name).lower() if name is not None else ""
        if name_str:  # Only add non-empty names
            drug_names_lower[name_str] = drug_id

    all_drug_names = list(drug_names_lower.keys())

    for text in extracted_texts:
        text_clean = str(text).lower().strip()

        # Skip very short text (likely noise)
        if len(text_clean) < 3:
            continue

        # Try exact match first
        if text_clean in drug_names_lower:
            drug_id = drug_names_lower[text_clean]
            if drug_id not in matched_drugs and drug_id in drug_map:
                matched_drugs.append(drug_id)
                continue

        # Try fuzzy matching (find close matches)
        close_matches = get_close_matches(text_clean, all_drug_names, n=1, cutoff=0.75)
        if close_matches:
            matched_name = close_matches[0]
            drug_id = drug_names_lower[matched_name]
            if drug_id not in matched_drugs and drug_id in drug_map:
                matched_drugs.append(drug_id)

    return matched_drugs


# DDI Type mapping — loaded once at startup
with open(DDI_TYPE_MAP, 'r') as f:
    ddi_type_map = json.load(f)

def get_type_desc(ddi_type):
    return ddi_type_map.get(str(ddi_type), "Unknown interaction type")


# 6 PREDICTION FUNCTION
def predict_interaction(model, graph_data, drug_a_id, drug_b_id, threshold=DEFAULT_THRESHOLD, device='cpu'):
    """
    Predict drug-drug interaction using the trained model.
    
    FIXED: Now matches the notebook's predict_ddi_multi() logic exactly.
    """
    model.eval()
    drug_map = graph_data['drug_map']

    # Check if drugs exist
    if drug_a_id not in drug_map or drug_b_id not in drug_map:
        return {
            'error': True,
            'message': f"Drug ID not found in database. Available drugs: {len(drug_map)}"
        }

    # Get node indices
    u = drug_map[drug_a_id]
    v = drug_map[drug_b_id]
    query_node_indices = torch.tensor([u, v], dtype=torch.long)

    # Extract 1-hop subgraph (same as training)
    from torch_geometric.utils import k_hop_subgraph

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=query_node_indices,
        num_hops=1,
        edge_index=graph_data['edge_index'],
        relabel_nodes=True
    )
    subgraph_u = mapping[0]
    subgraph_v = mapping[1]
    query_edge = torch.tensor([[subgraph_u], [subgraph_v]], device=device)

    # Get features
    x_v1 = graph_data['x_v1'][subset].to(device)
    x_v2 = graph_data['x_v2'][subset].to(device)
    x_v3 = graph_data['x_v3'][subset].to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        # Get predictions
        logits_bin, logits_type = model(x_v1, x_v2, x_v3, edge_index, query_edge)
        
        # Binary prediction (FIXED: Use logits directly, not probs)
        probs_bin = torch.softmax(logits_bin, dim=1)[0]
        prob_interaction = probs_bin[1].item()
        binary_pred = 1 if prob_interaction > threshold else 0
        
        # Type prediction
        probs_type = torch.softmax(logits_type, dim=1)[0]
        type_idx = torch.argmax(logits_type, dim=1).item()
        type_prob = probs_type[type_idx].item()
        
        # Get type name
        try:
            type_name = graph_data['encoder'].inverse_transform([type_idx])[0]
        except:
            type_name = str(type_idx)

    return {
        'error': False,
        'binary_prediction': binary_pred,
        'interaction_probability': prob_interaction,
        'predicted_type': type_name,
        'type_probability': type_prob,
        'drug_a': drug_a_id,
        'drug_b': drug_b_id,
        'threshold_used': threshold  # Track which threshold was used
    }


# INITIALIZATION
try:
    model, drug_map, smiles_df, name_dict, config, graph_data = load_system()
    smiles_dict = dict(zip(smiles_df['drug_id'], smiles_df['smiles']))
    # Load metadata for display
    with open(MODEL_CONFIG, 'r') as f:
        metadata_list = json.load(f)
    metadata = metadata_list[0] if isinstance(metadata_list, list) else metadata_list

except Exception as e:
    st.error(f"✘ Initialization Failed: {e}")
    st.stop()


# Helper for the searchable dropdown
def get_label(drug_id):
    name = name_dict.get(drug_id, "Unknown")
    return f"{name} ({drug_id})"

st.markdown("""
<style>
.stApp{background-color:#262624;}
.block-container{max-width:1300px;padding:3rem;}
section[data-testid="stSidebar"]{background-color:#262624!important;border-right:1px solid #dbcfcc;}

.sidebar-heading{color:#b06045;font-size:1.4rem;font-weight:600;padding-bottom:.4rem;margin-bottom:2rem;border-bottom:1px solid #dbcfcc;}
.main-header{font-size:2rem;color:#b06045;text-align:center;font-weight:800;margin-bottom:0;}
.paragraph{font-size:1rem;color:#dbcfcc;text-align:center;padding-bottom:.5rem;}
.section-heading{color:#cb785c;font-size:1.3rem;font-weight:600;padding-top:2rem;}
.drug-title{text-align:center;font-size:1.1rem;font-weight:700;color:#666;}

div[data-baseweb="select"]>div{background-color:#ecebe3!important;border-radius:10px!important;border:none!important;}
div[data-baseweb="select"] input{color:#392d2b!important;caret-color:#392d2b!important;}
div[data-baseweb="select"] div{color:#392d2b!important;}
div[data-baseweb="select"] input::placeholder{color:#392d2b!important;opacity:.8!important;}

div[data-testid="stForm"]{border:none!important;background:transparent!important;padding:0!important;}
div[data-testid="stFormSubmitButton"]>button{background-color:#cb785c!important;color:#fff!important;border-radius:10px!important;border:none!important;font-weight:600;padding:.6rem 2.5rem;}
div[data-testid="stFormSubmitButton"]>button:hover{background-color:#b06045!important;}
div.stButton > button:first-child {background-color: #cb785c;color: white;width: 100%;font-weight: 600;border:none;}
div.stButton > button:first-child:hover{background-color: #b06045;}

div[data-testid="stVerticalBlockBorderWrapper"]{border:2px solid #D2691E!important;background-color:#fff!important;border-radius:12px;}
div[data-testid="stAlert"]{background-color:#ffebd6!important;color:#4e5e6c!important;border-radius:12px!important;border:none!important;}
div[data-testid="stAlert"] p{color:#4e5e6c!important;}

.footer{position:fixed;bottom:0;left:0;width:100%;background-color:#ecebe3;color:#4e5e6c;text-align:center;padding:10px;font-size:.9rem;z-index:1000;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">PharmGAT-v2: GNN-Powered Drug Safety Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="paragraph">PharmGAT-v2: Multi-View Graph Attention Network (GATv2) for real-time adverse drug-drug interaction (DDI) screening and mechanism classification. Custom-trained on <b>DrugBank</b> molecular datasets and patient clinical profiles. High precision, dynamic attention-driven, and clinically grounded.<br> <blockquote>PharmGAT-v2 Project — Research and Development by Pranj</blockquote></div>', unsafe_allow_html=True)

# --- SIDEBAR: PATIENT RISK PROFILING ---
with st.sidebar:
    st.markdown('<div class="sidebar-heading">Patient Risk Profile</div>', unsafe_allow_html=True)
    st.markdown("*Personalize interaction analysis based on patient CVD risk factors*")

    st.session_state.enable_risk_profiling = st.checkbox(
        "Enable Patient Risk Profiling",
        value=st.session_state.enable_risk_profiling)

    enable_risk_profiling = st.session_state.enable_risk_profiling

    # Initialize default values
    risk_level = "MEDIUM"
    risk_score = 0
    risk_factors = []

    if enable_risk_profiling:
        st.markdown('> Please Enter Your Details:')
        patient_age = st.slider("Age (years)", min_value=18, max_value=90, value=50, step=1)
        col1, col2 = st.columns(2)
        # Blood pressure
        with col1:
            bp_systolic = st.number_input("BP Systolic", min_value=80, max_value=220, value=120, step=5)
        with col2:
            bp_diastolic = st.number_input("BP Diastolic", min_value=50, max_value=140, value=80, step=5)
        # Cholesterol Level
        cholesterol = st.selectbox(
            "Cholesterol Level",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "High"}[x],
            index=0
        )
        # Lifestyle Factors
        smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker", index=0)
        activity = st.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: "Inactive" if x == 0 else "Active", index=1)

        # Calculate Risk and SAVE to session state
        st.session_state.risk_level, st.session_state.risk_score, st.session_state.risk_factors = calculate_patient_risk_score(
            patient_age, bp_systolic, bp_diastolic, cholesterol, smoking, activity
        )

        # Use local variables for display (read FROM session state)
        risk_level = st.session_state.risk_level
        risk_score = st.session_state.risk_score
        risk_factors = st.session_state.risk_factors

        # Display Risk Assessment
        st.markdown("### Risk Result:")
        risk_colors = {"LOW": "#0e992a", "MEDIUM": "#ff8c00", "HIGH": "#ce0000"}
        risk_color = risk_colors[risk_level]

        st.markdown(f"""<div style="border: 2px solid; border-color: {risk_color}; color: {risk_color}; padding: 5px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.3rem; margin-bottom:2rem;">{risk_level} RISK <br><span style="font-weight: normal; font-size: 0.9rem;">Risk Score: {risk_score}/100</span></div>""", unsafe_allow_html=True)

        # FIX 1: risk_factors display is now INDEPENDENT from population context block
        if risk_factors:
            st.markdown("### **Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.markdown("✅ No major risk factors detected.")

        # FIX 1: Population context ALWAYS shown regardless of risk_factors being empty
        cvd_df = load_cvd_population()
        pop = get_population_context(cvd_df, patient_age, bp_systolic, cholesterol, smoking, activity)

        if pop:
            st.markdown("### Dataset Population Context:")
            st.markdown(f"""
            <div style="background:#1a1a18;border-radius:8px;padding:10px;margin-top:0.5rem;font-size:0.95rem;color:#dbcfcc;">
            From <b>{pop['cohort_size']}</b> similar patients in the CVD dataset:<br><br>
            <span style="font-size:1.3rem;font-weight:700;color:{'#ce0000' if pop['cvd_rate']>50 else '#ff8c00' if pop['cvd_rate']>30 else '#0e992a'};">
            {pop['cvd_rate']:.1f}%</span> had a cardiovascular diagnosis.<br>
            <span style="font-size:0.9rem;color:#999;">Avg systolic BP in cohort: {pop['avg_systolic']:.0f} mmHg</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Enable to personalize drug interaction verdict based on patient cardiovascular risk factors.")


# MAIN PAGE
st.markdown('<div class="section-heading">Enter Drugs Pair OR Upload the Prescription Image</div>', unsafe_allow_html=True)

# Add tabs for different input methods
tab1, tab2 = st.tabs(["✎ Manual Entry", "➜] Upload Image"])

with tab1:
    st.markdown("### Enter drugs from the dropdown input list")
    if "drug_count" not in st.session_state:
        st.session_state.drug_count = 2

    MAX_DRUGS = 4

    with st.form("drug_form"):
        total_columns = st.session_state.drug_count + (
            1 if st.session_state.drug_count < MAX_DRUGS else 0
        )
        cols = st.columns(total_columns)
        selected_drugs = []

        for i in range(st.session_state.drug_count):
            with cols[i]:
                drug = st.selectbox(
                    f"Drug {i+1}",
                    list(drug_map.keys()),
                    format_func=get_label,
                    key=f"drug_{i}"
                )
                selected_drugs.append(drug)

        col_add, col_predict = st.columns([1, 4])
        with col_add:
            add_clicked = st.form_submit_button("十 Drug")
        with col_predict:
            predict_clicked = st.form_submit_button("PREDICT")

    if add_clicked and st.session_state.drug_count < MAX_DRUGS:
        st.session_state.drug_count += 1
        st.rerun()

    if predict_clicked:
        # Remove duplicates
        unique_drugs = list(dict.fromkeys(selected_drugs))

        if len(unique_drugs) < 2:
            st.warning("⚠︎ Please select at least two distinct drugs.")
            st.stop()
        else:
            pairs = list(itertools.combinations(unique_drugs, 2))
            
            for drug_a, drug_b in pairs:
                # Determine threshold based on patient risk profiling
                if st.session_state.enable_risk_profiling:
                    threshold = get_risk_adjusted_threshold(st.session_state.risk_level)
                else:
                    threshold = DEFAULT_THRESHOLD
                    
                with st.spinner(f"Analyzing {name_dict.get(drug_a)} + {name_dict.get(drug_b)} ..."):
                    result = predict_interaction(
                        model,
                        graph_data,
                        drug_a,
                        drug_b,
                        threshold=threshold,
                        device='cpu'
                    )

                if result.get('error'):
                    st.error(f"✘ {result['message']}")
                    continue

                name_a = name_dict.get(drug_a, drug_a)
                name_b = name_dict.get(drug_b, drug_b)

                # Get drug descriptions
                drug_a_desc = get_drug_description(name_a)
                drug_b_desc = get_drug_description(name_b)

                score = result['interaction_probability']
                safe_prob = 1 - score
                binary_pred = result['binary_prediction']
                pred_type = result['predicted_type']
                type_prob = result['type_probability']

                # Generate Molecular Images
                mol_a = Chem.MolFromSmiles(smiles_dict.get(drug_a, ""))
                mol_b = Chem.MolFromSmiles(smiles_dict.get(drug_b, ""))

                img_a = Draw.MolToImage(mol_a, size=(500, 500)) if mol_a else None
                img_b = Draw.MolToImage(mol_b, size=(500, 500)) if mol_b else None

                # PREDICTION RESULT
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1, 1.5, 1])
                    with col1:
                        st.markdown(f'<div class="paragraph">{name_a}</div>', unsafe_allow_html=True)
                        if img_a:
                            st.image(img_a, width='stretch')
                    with col2:
                        text_color = "#c41414" if binary_pred == 1 else "#0e992a"
                        status_text = "⚠︎ ADVERSE INTERACTION DETECTED " if binary_pred == 1 else "(✔) SAFE PAIR, NO INTERACTION DETECTED"
                        items = [
                            f'<li style="font-weight:600;font-size:1.14rem;margin-bottom:8px;color:{text_color};">{status_text}</li>',
                            f'<li>Drug Pair: <b>{name_a} + {name_b}</b></li>',
                        ]

                        if binary_pred == 1:
                            items.append(f'<li>Interaction Probability: <b>{score:.2%}</b></li>')
                            ddi_type_desc = get_type_desc(pred_type)
                            items.append(f'<li>Interaction Side Effect: <b>{ddi_type_desc}</b></li>')
                            items.append(f'<li>Interaction Type {pred_type} Probability: <b>{type_prob:.2%}</b></li>')
                        else:
                            items.append(f'<li>Safe Probability: <b>{safe_prob:.2%}</b></li>')

                        items.append(f"<li>{name_a} Description: <b>{drug_a_desc}</b></li>")
                        items.append(f"<li>{name_b} Description: <b>{drug_b_desc}</b></li>")

                        # FIX 2: Added LOW risk verdict case — all three tiers now handled
                        if st.session_state.enable_risk_profiling and binary_pred == 1:
                            if st.session_state.risk_level == "HIGH":
                                items.append("<li style='color:#ce0000;'>Personalized Verdict: <b>⚠ HIGH-RISK PATIENT:</b> Extra caution required — consult cardiologist before prescribing.</li>")
                            elif st.session_state.risk_level == "MEDIUM":
                                items.append("<li style='color:#ff8c00;'>Personalized Verdict: <b>⚠ MEDIUM-RISK PATIENT:</b> Monitor closely — consider dose adjustment.</li>")
                            elif st.session_state.risk_level == "LOW":
                                items.append("<li style='color:#0e992a;'>Personalized Verdict: <b>ℹ LOW-RISK PATIENT:</b> Interaction detected — standard clinical monitoring advised.</li>")

                        html = (
                            '<div style="padding:0.5rem;">'
                            '<ul style="font-size:1.05rem;line-height:1.6;padding-left:1.2rem;">'
                            + "".join(items)
                            + "</ul></div>"
                        )
                        st.markdown(html, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="paragraph">{name_b}</div>', unsafe_allow_html=True)
                        if img_b:
                            st.image(img_b, width='stretch')

with tab2:
    uploaded_file = st.file_uploader(
        "Choose prescription image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of the prescription"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, width=300)

        with col2:
            if st.button("Extract Drugs from Image"):
                with st.spinner("Analyzing prescription image..."):
                    # Load OCR reader
                    reader = load_ocr_reader()

                    if reader is not None:
                        # Extract text
                        extracted_texts = extract_text_from_image(image, reader)

                        if extracted_texts:
                            st.success(f"✓ Extracted {len(extracted_texts)} text elements")

                            # Match to drug database
                            matched_drugs = match_drugs_from_text(extracted_texts, name_dict, drug_map)

                            if matched_drugs:
                                st.success(f"✓ Found {len(matched_drugs)} drugs in database")

                                # Store in session state
                                st.session_state.scanned_drugs = matched_drugs

                                # Display matched drugs
                                st.markdown("**Detected Drugs:**")
                                for drug_id in matched_drugs:
                                    drug_name = name_dict.get(drug_id, drug_id)
                                    st.markdown(f"• {drug_name} ({drug_id})")
                            else:
                                st.warning("⚠︎ No matching drugs found in database. Try manual entry.")
                        else:
                            st.error("✘ No text detected. Please upload a clearer image.")

    # Show scanned drugs if available
    if 'scanned_drugs' in st.session_state and st.session_state.scanned_drugs:

        if st.button(" PREDICT", key="predict_scanned"):
            # Use scanned drugs for prediction
            selected_drugs = st.session_state.scanned_drugs

            if len(selected_drugs) < 2:
                st.warning("⚠︎ Need at least 2 drugs for interaction analysis")
            else:
                pairs = list(itertools.combinations(selected_drugs, 2))

                for drug_a, drug_b in pairs:
                    # Determine threshold
                    if st.session_state.enable_risk_profiling:
                        threshold = get_risk_adjusted_threshold(st.session_state.risk_level)
                    else:
                        threshold = DEFAULT_THRESHOLD

                    with st.spinner(f"Analyzing {name_dict.get(drug_a)} + {name_dict.get(drug_b)}..."):
                        result = predict_interaction(
                            model, graph_data, drug_a, drug_b, threshold=threshold, device='cpu'
                        )

                    if result.get('error'):
                        st.error(f"✘ {result['message']}")
                        continue

                    # Display results (same as manual entry)
                    name_a = name_dict.get(drug_a, drug_a)
                    name_b = name_dict.get(drug_b, drug_b)

                    drug_a_desc = get_drug_description(name_a)
                    drug_b_desc = get_drug_description(name_b)

                    score = result['interaction_probability']
                    safe_prob = 1 - score
                    binary_pred = result['binary_prediction']
                    pred_type = result['predicted_type']
                    type_prob = result['type_probability']

                    mol_a = Chem.MolFromSmiles(smiles_dict.get(drug_a, ""))
                    mol_b = Chem.MolFromSmiles(smiles_dict.get(drug_b, ""))
                    img_a = Draw.MolToImage(mol_a, size=(500, 500)) if mol_a else None
                    img_b = Draw.MolToImage(mol_b, size=(500, 500)) if mol_b else None

                    with st.container(border=True):
                        col1, col2, col3 = st.columns([1, 1.5, 1])
                        with col1:
                            st.markdown(f'<div class="paragraph">{name_a}</div>', unsafe_allow_html=True)
                            if img_a:
                                st.image(img_a, width='stretch')
                        with col2:
                            text_color = "#c41414" if binary_pred == 1 else "#0e992a"
                            status_text = "⚠︎ ADVERSE INTERACTION DETECTED" if binary_pred == 1 else "✓ SAFE PAIR"

                            items = [
                                f'<li style="font-weight:600;font-size:1.14rem;margin-bottom:8px;color:{text_color};">{status_text}</li>',
                                f'<li>Drug Pair: <b>{name_a} + {name_b}</b></li>',
                            ]

                            if binary_pred == 1:
                                items.append(f'<li>Interaction Probability: <b>{score:.2%}</b></li>')
                                ddi_type_desc = get_type_desc(pred_type)
                                items.append(f'<li>Interaction Side Effect: <b>{ddi_type_desc}</b></li>')
                                items.append(f'<li>Type Confidence: <b>{type_prob:.2%}</b></li>')
                                items.append(f'<li>Modify Therapy/Monitor Closely</li>')
                            else:
                                items.append(f'<li>Safe Probability: <b>{safe_prob:.2%}</b></li>')
                                items.append(f'<li>Note: Even if no interactions are found, monitor it and consider patient individual allergies.</li>')

                            items.append(f"<li>{name_a} Description: <b>{drug_a_desc}</b></li>")
                            items.append(f"<li>{name_b} Description: <b>{drug_b_desc}</b></li>")

                            # FIX 2: Added LOW risk verdict case — all three tiers now handled
                            if st.session_state.enable_risk_profiling and binary_pred == 1:
                                if st.session_state.risk_level == "HIGH":
                                    items.append("<li style='color:#ce0000;'>Personalized Verdict: <b>⚠ HIGH-RISK PATIENT:</b> Extra caution required — consult cardiologist before prescribing.</li>")
                                elif st.session_state.risk_level == "MEDIUM":
                                    items.append("<li style='color:#ff8c00;'>Personalized Verdict: <b>⚠ MEDIUM-RISK PATIENT:</b> Monitor closely — consider dose adjustment.</li>")
                                elif st.session_state.risk_level == "LOW":
                                    items.append("<li style='color:#0e992a;'>Personalized Verdict: <b> LOW-RISK PATIENT:</b> Interaction detected — standard clinical monitoring advised.</li>")

                            html = '<div style="padding:0.5rem;"><ul style="font-size:1.05rem;line-height:1.6;padding-left:1.2rem;">' + "".join(items) + "</ul></div>"
                            st.markdown(html, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="paragraph">{name_b}</div>', unsafe_allow_html=True)
                            if img_b:
                                st.image(img_b, width='stretch')


# Footer
st.markdown(
    f"""<div class='footer'>● Model: PharmGAT-v2 ({config['hidden_dim']}D, {config['n_gat_layers']} layers) ● Research Version: For Academic & Educational Purposes Only
    </div>""", unsafe_allow_html=True)