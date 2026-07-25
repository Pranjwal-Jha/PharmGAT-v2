- ✅ ChCh-Miner: | graph structure | Who interacts with whom

Topology (The Map): `ChCh-Miner` provides the global network of how drugs connect. This helps the model learn "Guilt by Association" (e.g., if Drug A connects to Drug B, and Drug B causes heart issues, Drug A might too).

- ✅ ddis1.csv: | Label (Supervision) | What interaction exists

Ground Truth (The Teacher): ddis1.csv contains both positive examples (bad interactions) and negative examples (safe pairs). This is critical for the model to learn the difference.
- ✅ drug_smiles1.csv: chemistry/Node features | chemical mechanism

Chemical Identity (The DNA): `drug_smiles1.csv` allows you to generate "Node Features." Without this, your nodes are just blank circles. With this, your nodes contain actual chemical intelligence (benzene rings, hydroxyl groups, etc.).


### 1. Understanding the `DBXXXXX` Codes
Those codes are DrugBank Accession Numbers. They are unique identifiers for every drug molecule.

### 2. `ChCh-Miner_durgbank-chem-chem.tsv`
- B00862, DB00966, etc. are DrugBank IDs
- Each line = one known drug–drug interaction
- This file is a pure edge list

It only says: “These two drugs are known to interact (somehow).”<br>
This does NOT tell: What the interaction is, How severe it is, and What side effect occurs.

### 3. `ddis.csv`
- d1, d2: DrugBank drug pair
- type: Interaction label
- Neg samples: Drugs known not to interact