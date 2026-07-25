# Complete Guide: Graph Attention Networks v2 for Drug-Drug Interaction Prediction

This project implements a state-of-the-art Graph Attention Network (GATv2) for predicting cardiovascular drug-drug interactions (DDI). The model learns from molecular structures (SMILES) and known interaction patterns to predict interaction types between drug pairs.

## Part (1): Understanding the GAT v2 Fundamentals

### 1a. What is a Graph Neural Network (GNN)?


| Feature | Traditional Neural Networks | Graph Neural Networks (GNNs) |
| :--- | :--- | :--- |
| **Data Type** | Grid-like data (Images, sequences) | Graph-structured data |
| **Structure** | Fixed structure and size | Variable structure and size |
| **Examples** | CNN (Images), RNN (Text) | Social networks, molecules, drug interactions |

### 1b. Architecture

#### What is a Graph Convolution `GATv2Conv Layers`?
Think of an image. In traditional CNNs, we slide a filter over a 2D grid of pixels. But what about data that isn't a grid? Social networks, molecules, road networks - these are graphs where connections are irregular.
```
Traditional CNN:           Graph Neural Network:
┌───┬───┬───┐              
│ 1 │ 2 │ 3 │                (A)───(B)
├───┼───┼───┤                 │     │
│ 4 │ 5 │ 6 │                (C)───(D)───(E)
├───┼───┼───┤                       │
│ 7 │ 8 │ 9 │                      (F)
└───┴───┴───┘              
Fixed grid neighbors       Irregular neighbors
```
> Graph Convolution = Instead of fixed filters, we aggregate information from neighbors.


#### GATv2 vs GAT 
**Original GAT (2018) - Static attention**
The non-linearity is applied after the nodes are combined and reduced by the weight vector $a^T$
$$ \text{attention} = \text{LeakyReLU}\left(\mathbf{a}^T \cdot \left[\mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j\right]\right) $$

*Problem:* For fixed $h_i$, attention to all neighbors is computed BEFORE seeing $h_j$

**GATv2 (2021) - Dynamic attention**
By applying the non-linearity to the concatenated features before the dot product, the model achieves **dynamic attention**. In standard GAT, the ranking of importance for neighbors is "static" (fixed for all query nodes), but in this version, the importance can change based on the specific query node $h_i$
$$ \text{attention} = \mathbf{a}^T \cdot \text{LeakyReLU}\left(\mathbf{W} \cdot \left[\mathbf{h}_i \, \Vert \, \mathbf{h}_j\right]\right) $$

_Fix:_ LeakyReLU is INSIDE, so attention depends on BOTH nodes dynamically

#### Optimal Layers in GATv2
Each layer aggregates information from 1-hop neighbors:

- Layer 1: Node A learns about direct neighbors (B, C, D)
- Layer 2: Node A learns about 2-hop (neighbors of neighbors)  
- Layer 3: Node A learns about 3-hop (3 edges away)

```
           Layer 0        Layer 1         Layer 2         Layer 3
             (A)      →   (A) knows   →   (A) knows   →   (A) knows
              │           B,C,D           entire           entire
           B──C──D                        local            subgraph
              │                           region
              E
```
_**Trade-off:**_

- More layers = More context = Better understanding of graph structure
- More layers = More memory = Risk of over-smoothing (all nodes become similar)
> For molecular graphs, `2-4` layers is typically optimal.
#### What is Attention?
Not all neighbors are equally important! Attention lets the model learn which neighbors matter more.

_Example:_ In Project Drug-Drug Interaction graph:

- Drug A interacts with Drugs B, C, D, E
- But Drug B might have a much stronger influence on A's behavior than D
- Attention learns these importance weights automatically

_**The Problem with Single Attention:**_ One attention mechanism learns ONE way to weight neighbors. But relationships can be complex!

**Example:** Drug A interacting with Drug B might be important because:

1. They share similar molecular structure
2. They affect the same enzyme
    3. They have opposite solubility properties

_**Multi-Head Attention Solution:**_ Run K independent attention mechanisms in parallel, then combine:
```
                    ┌─── Head 1 (learns structural similarity) ───┐
                    │                                             │
Input Features ─────┼── Head 2 (learns enzyme interaction)   ─────┼──→ Concatenate/Average
                    │                                             │
                    └─── Head 3 (learns solubility patterns)  ────┘
```
8 heads, each produces 32-dim output
$head_1$ = $attention_1(x)$  # Shape: [N, 32]
$head_2$ = $attention_2(x)$  # Shape: [N, 32]
...
$head_8$ = $attention_8(x)$  # Shape: [N, 32]

> Concatenate: [N, 32*8] = [N, 256]
output = concat(head_1, head_2, ..., head_8)

| Heads | Trade-off | Common Use Cases |
| :--- | :--- | :--- |
| **2–4** | Faster, less expressive | Small datasets, resource-constrained GNNs |
| **8** | **Sweet spot** for most tasks | Original [Transformer](https://arxiv.org) & [GAT](https://arxiv.org) paper |
| **12–16** | Higher complexity, better nuance | [BERT-Base](https://huggingface.co) (12) and [BERT-Large](https://huggingface.co) (16) |
| **64+** | Diminishing returns, expensive | Massive LLMs (e.g., [GPT-3](https://arxiv.org) uses 96 heads) |

> Our model: `8 heads × 32 dims = 256` output dimensions per layer

#### What is Hidden Dimension?
It's the representational capacity of each layer - how many numbers describe each node.
- Dataset drug features: 1032 dimensions (fingerprints + descriptors)
- Hidden layer: 256 dimensions

Think of it as compression:
```
Input:  [1032 numbers describing drug A]
                    ↓
Layer 1: [256 numbers - learned abstract representation]
                    ↓
Layer 2: [256 numbers - even more abstract]
                    ↓
Layer 3: [128 numbers - final node embedding]
```
| Dimension | Description |
| :--- | :--- |
| **32-64** | Lightweight, faster training, may underfit |
| **128-256** | **Standard** for molecular/biomedical graphs |
| **512-1024** | Large models (protein folding, AlphaFold) |
| **4096+** | LLMs (GPT-4, Claude) |

**Memory Formula**

$
\begin{aligned}
\text{Layer Memory} &\approx n_{\text{nodes}} \times n_{\text{heads}} \times \text{hidden\_dim} \times \text{sizeof(float16)} \\
&= 1706 \times 8 \times 256 \times 2 \text{ bytes} \\
&\approx 7 \text{ MB per layer (activations)} \\
&+ 7 \text{ MB (gradients)} \\
&+ 14 \text{ MB (Adam optimizer states)} \\
&\approx 28 \text{ MB per layer} \times 3 \text{ layers} \approx 84 \text{ MB}
\end{aligned}
$

#### 1-Hop Neighbor Expansion (The Memory Killer!)

In GNNs, computing the representation for a single node requires information from its \(k\)-hop neighborhood. As you add layers, the number of required nodes grows exponentially.

**The Geometry of Expansion**
When you batch 800 edges, you aren't just processing those edges; you are processing their entire "influence zone.
- Initial Batch: Edges \((A\rightarrow B),(C\rightarrow D),(A\rightarrow E)\)
- Direct Nodes: \(\{A,B,C,D,E\}\) (5 unique nodes)
- The 1-Hop Expansion: To compute attention for node \(A\), you must pull its neighbors: \(\{B,C,F,G,H\}\)
- The 2-Hop Expansion: To compute the neighbors' values, you now need their neighbors (e.g., \(F\)'s friends). 

> To prevent the "Memory Killing" by k-Hop Neighbor Expansion and prevent from crashing GPU, industry-standard models (like **GraphSAGE**) use **Neighbor Sampling**:
```
                              Layer 3 needs:
                              ALL these nodes!
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
         ┌────┴────┐          ┌─────┴────┐          ┌─────┴────┐
         │ 27 nodes│          │ 45 nodes │          │ 32 nodes │
         └────┬────┘          └────┬─────┘          └────┬─────┘
              │                    │                     │
              │    Layer 2 needs ALL these              │
              └──────────┬─────────┴─────────────┬──────┘
                         │                       │
                    ┌────┴────┐             ┌────┴────┐
                    │ 9 nodes │             │ 7 nodes │
                    └────┬────┘             └────┬────┘
                         │     Layer 1           │
                         └──────────┬────────────┘
                                    │
                              ┌─────┴────┐
                              │  Target  │
                              │   Node   │
                              └──────────┘

Total nodes needed for ONE target node: 1 + 16 + 104 = 121 nodes!
```
**The Math of Explosion:** If average node degree is d (average neighbors per node):
| Layer | Nodes Needed | With d=50 |
| :--- | :--- | :--- |
| Target| 1 | 1 |
| 1-hop | d | 50 |
| 2-hop | d² | 2,500 |
| 3-hop | d³ | 125,000 |
| Total | O(d³) | ~127,000 |

For DDI graph:
- 1,706 nodes, 383,616 edges
- Average degree = 383,616 / 1,706 ≈ 225 neighbors per drug!
- 3-layer GNN needs: 225³ = 11.4 million nodes (but you only have 1,706!)
> This means for a 3-layer GNN, computing ONE node's embedding requires loading the entire graph multiple times.

#### Subgraph Edge Index
PyTorch Geometric stores graphs as two things:
- Node features: x with shape [N, F] (N nodes, F features)
- Edge index: which nodes connect to which
```
# Edge index format: [2, E] where E = number of edges
edge_index = tensor([[0, 0, 1, 2, 3],   # Source nodes
                     [1, 2, 0, 3, 2]])  # Target nodes

# This represents edges: 0→1, 0→2, 1→0, 2→3, 3→2
#
#    (0)─────(1)
#     │       
#     │       
#    (2)─────(3)
```

#### What is Adjacency Matrix?
A matrix representation of graph connectivity:
```
# For 4 nodes with edges: 0→1, 1→2, 2→3
adj_matrix = [
    [0, 1, 0, 0],  # Node 0 connects to Node 1
    [0, 0, 1, 0],  # Node 1 connects to Node 2
    [0, 0, 0, 1],  # Node 2 connects to Node 3
    [0, 0, 0, 0],  # Node 3 connects to nothing
]
```
| Type | Storage Logic | Your Graph (1,706 nodes) | Complexity |
| :--- | :--- | :--- | :--- |
| **Dense** | Store **ALL** entries (including zeros) | $1706 \times 1706 = 2.9\text{M}$ entries $\approx 12\text{ MB}$ | $O(N^2)$ |
| **Sparse** | Store **ONLY** existing edges | $383,616$ edges $\approx 3\text{ MB}$ | $O(E)$ |

#### GraphSAGE
GraphSAGE = Graph SAmple and aggreGatE [Stanford 2017]

Core Idea: Fixed-Size Neighbor Sampling. Instead of using ALL neighbors, randomly sample a fixed number K at each layer.
```
Original Graph:                    GraphSAGE Sampled (K=2 per layer):

    (F)──(C)──(D)                     (F)      (D)
     │    │    │                       │        │
    (G)──(A)──(E)        ───────►     (G)──(A)──(E)
     │    │    │          Sample            │    
    (H)──(B)──(I)                          (B)──(I)
          │
         (J)

Target: A                          Target: A
Neighbors of A: B,C,E,G (4)        Sampled: E,G (K=2)
Layer 2 nodes: D,F,H,I,J,C,B,E,G   Sampled: B,I,D,F (K=2 each)

Full graph: 10 nodes               Sampled: 7 nodes (30% reduction)
```
### 1c. How GAT Works

**Step 1: Node Features**
```python
Each drug has features:
- Molecular fingerprint (512 bits)
- Molecular weight
- LogP (lipophilicity)
- H-bond donors/acceptors
- etc.
```

**Step 2: Attention Computation**
```
For each drug pair (i, j):
1. Transform features: h'_i = W * h_i
2. Compute attention score: e_ij = LeakyReLU(a^T [h'_i || h'_j])
3. Normalize: α_ij = softmax(e_ij)
```

**Step 3: Aggregation**
```
Update drug representation:
h_i^new = Σ α_ij * h'_j (sum over neighbors)
```

**Step 4: Multi-head Attention**
```
Run attention K times (K heads):
- Head 1: Captures pharmacokinetic interactions
- Head 2: Captures pharmacodynamic interactions
- Head 3: Captures structural similarities
- etc.

Concatenate all heads: h_i^final = [head1 || head2 || ... || headK]
```
#### Why GATv2 for DDI?

✅ **Captures Graph Structure**: Drugs form a network of interactions  
✅ **Attention Mechanism**: Identifies important drug relationships  
✅ **Molecular Features**: Leverages chemical structure information  
✅ **Multi-class Prediction**: Handles multiple interaction types  

#### Why Graphs for Drug Interactions?

```
Traditional Approach:
Drug A + Drug B → Interaction? (isolated pairs)

Graph (GAT) Approach:
Drug A ← connected to → Drug B
  ↓                        ↓
Drug C ← connected to → Drug D
(learns from network patterns)
```

**Advantages**:
1. **Network Effects**: Learns from similar drug pairs
2. **Transitive Relationships**: If A interacts with B, and B with C, what about A and C?
3. **Molecular Similarity**: Similar drugs have similar interaction patterns
4. **Multi-hop Reasoning**: Considers indirect relationships

### 1d. How GAT Works for DDI

1. **Node Embedding Learning**:
   - Each drug is represented by its molecular features
   - GAT layers learn to aggregate information from neighboring drugs
   - Attention mechanism focuses on most relevant drug interactions

2. **Attention Mechanism**:
   - Computes attention scores between drug pairs
   - Higher attention = stronger relationship
   - Multi-head attention captures different interaction patterns

3. **Edge Prediction**:
   - Combines embeddings of two drugs
   - MLP predicts interaction type
   - Handles 86 different interaction categories

### 1e Key Concepts

#### Graph Attention Networks (GAT)
- **Attention Mechanism**: Learns importance of neighboring nodes
- **Multi-head Attention**: Captures different relationship types
- **Masked Attention**: Only considers connected nodes
- **Permutation Invariant**: Order of nodes doesn't matter

#### Drug-Drug Interactions
- **Pharmacokinetic**: Affects drug absorption, distribution, metabolism, excretion
- **Pharmacodynamic**: Affects drug action at target site
- **Severity Levels**: Minor, moderate, major, contraindicated
- **Clinical Significance**: Impact on patient outcomes

#### Molecular Fingerprints
- **Morgan Fingerprints**: Circular fingerprints based on atom neighborhoods
- **Bit Vectors**: Binary representation of molecular features
- **Similarity**: Similar molecules have similar fingerprints
- **Substructure**: Captures presence of chemical motifs
---

## Part (2): DDI Dataset

### 2a Dataset Structure

**1. Drug-Drug Interactions (ddis.csv)**
- **File**: `dataset/drugdata/ddis.csv`
- **Total Interactions**: 191,808 drug pairs
- **Unique Drugs**: 1,706 drugs
- **Interaction Types**: 86 different interaction categories
- **Format**: `d1, d2, type, Neg samples`
```
d1        d2        type  Neg samples
DB04571   DB00460   0     DB01579$t
DB00855   DB00460   0     DB01178$t
...
```

- **d1, d2**: Drug pair (DrugBank IDs)
- **type**: Interaction type (0-85, 86 categories)
- **Neg samples**: Negative samples for training

**2. Drug SMILES (drug_smiles.csv)**
- **File**: `dataset/drugdata/drug_smiles.csv`
- **Total Drugs**: 1,706 drugs with SMILES representations
- **Format**: `drug_id, smiles`
- **Coverage**: 100% overlap with DDI data
```
drug_id   smiles
DB04571   CC1=CC2=CC3=C(OC(=O)C=C3C)C(C)=C2O1
DB00855   NCC(=O)CCC(O)=O
...
```

- **drug_id**: DrugBank ID
- **smiles**: Chemical structure representation

### 2b Data Split
- **Training**: 70% (134,265 edges)
- **Validation**: 15% (28,771 edges)
- **Testing**: 15% (28,772 edges)

### 2c Interaction Types

Dataset has **86 different interaction types**. Examples:
- Type 0: No significant interaction
- Type 1: Minor interaction
- Type 48: Most common (60,751 cases)
- Type 46: Second most common (34,360 cases)
- etc.

**Challenge**: Highly imbalanced classes!
- Some types have 60,000+ examples
- Some types have only 6-10 examples

---

## Part (3): Model Architecture Explained

### 3a Overall Pipeline

```
SMILES → Feature Extraction → Graph Construction → GAT → Prediction
```
### Feature Extraction
**Molecular Features from SMILES:**
- **Morgan Fingerprints**: 512-bit circular fingerprints (radius=2)
- **Molecular Descriptors**:
  - Molecular Weight (MW)
  - LogP (lipophilicity)
  - H-bond donors/acceptors
  - Topological Polar Surface Area (TPSA)
  - Rotatable bonds
  - Aromatic rings
  - Fraction of sp3 carbons

**Total Feature Dimension**: 520 (512 + 8)

#### Detailed Architecture

**1. Feature Extraction (RDKit)**
```python
Input: SMILES string
       ↓
Parse molecule
       ↓
Extract fingerprint (512 bits)
       ↓
Calculate descriptors (8 values)
       ↓
Output: 520-dimensional vector
```
#### GAT Architecture

```
Input: Drug Features (520-dim)
    ↓
GAT Layer 1 (8 heads, 256-dim) + ELU + Dropout
    ↓
GAT Layer 2 (8 heads, 256-dim) + ELU + Dropout
    ↓
GAT Layer 3 (8 heads, 128-dim) + ELU + Dropout
    ↓
Edge Embedding (concatenate source + target)
    ↓
MLP (256 → 128 → 64 → 86)
    ↓
Output: Interaction Type Prediction
```

**Graph Construction**
```python
Nodes: 1,706 drugs
Edges: 191,808 interactions (bidirectional = 383,616)
Node features: 520-dim vectors
Edge labels: Interaction types (0-85)
```

**GAT Layers**
```python
Layer 1: 520 → 256 (8 heads, concat)
         ↓ ELU + Dropout
Layer 2: 256 → 256 (8 heads, concat)
         ↓ ELU + Dropout
Layer 3: 256 → 128 (8 heads, concat)
         ↓ ELU + Dropout
```

**4. Edge Prediction**
$
\begin{aligned}
\text{1. Embeddings:} \quad & \mathbf{h}_i, \mathbf{h}_j \in \mathbb{R}^{128} \\
\text{2. Concatenate:} \quad & \mathbf{e}_{i,j} = [\mathbf{h}_i \, \Vert \, \mathbf{h}_j] \in \mathbb{R}^{256} \\
\text{3. MLP Layers:} \quad & \mathbf{e}_{i,j} \xrightarrow{\text{FC}} 128 \xrightarrow{\text{ELU}} 64 \xrightarrow{\text{FC}} 86 \\
\text{4. Output:} \quad & \mathbf{\hat{y}} = \text{Softmax}(\text{MLP}(\mathbf{e}_{i,j})) \in [0, 1]^{86}
\end{aligned}
$

## Part (4): Understanding Metrics

**Accuracy**
```
Accuracy = Correct Predictions / Total Predictions

Example: 75% accuracy means:
- Out of 100 drug pairs
- 75 interaction types predicted correctly
- 25 predicted incorrectly
```

**F1-Score**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Better for imbalanced data!
- Considers both false positives and false negatives
- Weighted F1: Accounts for class imbalance
```

**Per-Class Performance**
```
Type 48 (60,751 samples):
- Precision: 0.85 (85% of predictions are correct)
- Recall: 0.92 (92% of actual cases found)
- F1: 0.88

Type 41 (6 samples):
- Precision: 0.20 (only 20% correct)
- Recall: 0.17 (only 17% found)
- F1: 0.18 (poor performance due to few examples)
```


