Self-Healing Classification DAG using LangGraph and LLMs
Overview
This project implements a Self-Healing Classification DAG (Directed Acyclic Graph) using LangGraph and Large Language Models (LLMs). The system simulates an intelligent workflow that can detect, correct, and reclassify failed tasks automatically — making it “self-healing.”

It uses a multi-agent LangGraph workflow, where each node (agent) performs specific actions such as data validation, classification, logging, and error recovery.

🚀 Features
✅ Modular LangGraph-based architecture

✅ Self-healing mechanism for error recovery

✅ GPU-accelerated fine-tuning using PyTorch and Transformers

✅ Parameter-efficient fine-tuning (PEFT) supported

✅ Real-time progress logging using tqdm and rich

✅ High compatibility with CUDA-enabled devices

🧩 Project Structure
Self_Healing_Classification_DAG/ │

├── data/ # Input and training datasets

├── models/ # Fine-tuned / saved model weights

│── agents # Agent node implementations

│── dag_builder.py # DAG construction logic

│── trainer.py # Model fine-tuning script

│── evaluator.py # Evaluation and metrics

│── utils # Helper functions

│── init.py

├── requirements.txt # Dependencies

├── README.md # Project documentation

└── main.py # Entry point to run the workflow

⚙️ Installation Guide
🧾 1. Prerequisites

Python 3.10.x

VS Code (recommended)

CUDA-enabled GPU (NVIDIA)

pip (latest version)

🧠 2. Clone the Repository
git clone https://github.com/Kakumanu-Harshitha/Self_Healing_Classification_DAG.git
cd Self_Healing_Classification_DAG
🧰 3. Create a Virtual Environment
python -m venv self-healing
self-healing\Scripts\activate     # (Windows)
🔄 4. Verify GPU Setup
Before installing libraries, make sure CUDA is available:

nvidia-smi
If this shows your GPU details → proceed. Else, install proper NVIDIA drivers + CUDA Toolkit 11.8.

📦 5. Install Dependencies
Install all dependencies in one go:

pip install -r requirements.txt
If you face issues with PyTorch installation, run this manually first: ''' bash

pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118

##Then install other dependencies:

pip install -r requirements.txt --no-deps


# 📋 requirements.txt
```bash 
# GPU & Core Libraries
torch==2.2.2+cu118
torchvision==0.17.2+cu118
torchaudio==2.2.2+cu118
--index-url https://download.pytorch.org/whl/cu118

# Transformers & ML Stack
transformers==4.45.2
datasets==2.19.2
scikit-learn==1.4.2
accelerate==0.31.0
peft==0.11.1
bitsandbytes==0.43.3

# Utilities & Logging
python-json-logger==2.0.7
rich==13.7.1
tqdm==4.66.4
```
🔍 6. Verify Installation
After installing, check if Torch detects the GPU:

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
🧪 7. Run the Project

Start the system:

python main.py
Expected Output:
✅ DAG Initialized ✅ Agent_1 started classification... ⚠️ Error detected -> Self-healing triggered ✅ Error resolved. Reclassification successful. 🎉 All tasks completed successfully!

🧬 Fine-Tuning Script
The fine-tuning script trains your LLM or Transformer model (like bert-base-uncased or roberta-base) on your dataset using parameter-efficient fine-tuning (PEFT).

Example:

python src/trainer.py
--model_name bert-base-uncased
--train_file data/train.csv
--val_file data/val.csv
--output_dir models/self_healing_bert
--epochs 3
--batch_size 8

🧠 How It Works
DAG Initialization — LangGraph builds a graph with agent nodes.

Agent Execution — Each agent performs classification tasks.

Failure Detection — If an error occurs, it triggers a healing agent.

Self-Healing — The healing node re-evaluates and fixes misclassifications.

Result Aggregation — A judge node validates and finalizes the outcome.

📈 Future Enhancements
Integrate with LangChain ReAct Agents

Add LLM-based debate judge

Include visual DAG monitor

Implement real-time dashboard with Streamlit
