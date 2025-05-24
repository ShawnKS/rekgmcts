# ReKG-MCTS: Monte Carlo Tree Search for Knowledge Graph Reasoning with LLMs

Codebase for our ACL submission "ReKG-MCTS: Integrating Monte Carlo Tree Search with LLMs for Knowledge Graph Reasoning"

## 📖 Repository Overview

This repository implements a knowledge graph reasoning framework based on Monte Carlo Tree Search (MCTS) and Large Language Models (LLMs). The core idea is to:
1. **Use MCTS for structured search**: Explore paths on the knowledge graph.
2. **Leverage LLMs for semantic guidance**: Act as a policy model (selecting expansion directions) and a value evaluator (scoring path quality).
3. **Iterative optimization in four phases**: Node Selection → Path Expansion → Simulation → Value Backpropagation.

## 📂 Code Structure

### Core Modules

| File | Description |
|------|-------------|
| `mcts.py` | Core MCTS implementation<br>✅ Node definition (MCTSNode)<br>✅ Search process (MCTSPathFinder)<br>✅ UCB selection / Path expansion / Simulation backpropagation |
| `model.py` | LLM interface encapsulation<br>✅ HuggingFace model loading (HUGGINGFACE_LLM)<br>✅ GPT-4 API interface (GPT4_LLM)<br>✅ Answer generation and verification |
| `sparql_utils.py` | Knowledge graph interaction<br>✅ SPARQL query execution<br>✅ Entity and relation retrieval<br>✅ Freebase format processing |
| `utils.py` | Utility functions<br>✅ Answer extraction<br>✅ Relation cleaning<br>✅ Entity name resolution |
| `prompt.py` | Prompt template management<br>✅ Relation extraction prompts<br>✅ Path evaluation prompts<br>✅ Answer generation prompts |

## 🚀 Quick Start

### Environment Setup

1. **Install and Deploy Freebase Database**
**Requirements:**
- OpenLink Virtuoso 7.2.5 ([download](https://sourceforge.net/projects/virtuoso/files/virtuoso/))
- Python 3
- Freebase dump ([download](https://developers.google.com/freebase?hl=en))

**Setup Steps:**
```shell
# 1. Data Preprocessing
gunzip -c freebase-rdf-latest.gz > freebase  # Decompress dump (400G)
nohup python -u FilterEnglishTriplets.py 0<freebase 1>FilterFreebase 2>log_err &  # Clean data (output: 125G)

# 2. Virtuoso Installation
tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
cd virtuoso-opensource/database/
mv virtuoso.ini.sample virtuoso.ini

# 3. Start Virtuoso Service
../bin/virtuoso-t &  # Run in background
../bin/isql 1111 dba dba  # Connect to database

# 4. Import Data (Execute in isql console)
SQL> ld_dir('.', 'FilterFreebase', 'http://freebase.com');
SQL> rdf_loader_run();
```

2. **Configure SPARQL Endpoint**
```python
# Modify in sparql_utils.py
SPARQLPATH = "http://localhost:8890/sparql"  # Default Virtuoso endpoint
```

3. **Verify Database Connection**
```python
# Run the test script from Freebase Setup section
python test_sparql.py
```

4. **Configure model paths (e.g., Llama3)**
```python
# Configure model paths in main.py
MODEL_PATHS = {
    "Llama3-8B": "/path/to/Llama-3.8B",
}
```

### Running the Example

Basic execution command:
```bash
python main.py \
    --model Llama3-8B \
    --data cwq \
    --exploration_constant 0.5 \
    --depth 3 \
    --width 5 \
    --iterations 3 \
```

Key parameters:
- `--model`: Supported LLM types (Llama3-8B/70B, Qwen-7B, etc.)
- `--data`: Dataset selection (cwq/webqsp)
- `--exploration_constant`: UCB exploration coefficient
- `--depth`: Maximum mcts depth
- `--width`: Maximum mcts width
- `--iterations`: Maximum mcts iterations
