# Unifying AI Tutor Evaluation - Paper Replication

This repository contains the code and data for replicating the paper **"Unifying AI Tutor Evaluation"** (NAACL 2025). The project focuses on evaluating AI tutoring systems using the MRBench dataset with multiple language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🔍 Overview

This project implements a comprehensive evaluation framework for AI tutoring systems, specifically focusing on:

- **Mathematical dialogue evaluation** using MathDial prompts
- **Bridge dataset evaluation** using Bridge prompts
- **Multi-model inference** with Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.1
- **Automated evaluation metrics** for tutor response quality

## 📁 Project Structure

```
Unifying_AI_Tutor_Evaluation_PaperReplicate/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── code/                      # Source code
│   ├── inference.py           # Main inference script
│   ├── helper.py             # Utility functions for prompt formatting
│   └── data_analysis.ipynb   # Analysis and visualization notebook
├── data/                     # Dataset and prompts
│   ├── MRBench/
│   │   ├── MRBench_V1.json   # Main evaluation dataset
│   │   └── MRBench_V2.json   # Extended dataset
│   └── prompt/
│       ├── prompt_Bridge.txt  # Bridge evaluation prompts
│       └── prompt_MathDial.txt # MathDial evaluation prompts
└── paper/                    # Paper and results
    ├── 2025.naacl-long.57.pdf # Original paper
    └── paper_result.csv       # Reference results
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model inference)
- At least 16GB RAM (32GB recommended for larger models)

### Environment Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Unifying_AI_Tutor_Evaluation_PaperReplicate
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   # Test basic functionality
   python -c "import torch, transformers, helper; print('✓ All dependencies installed successfully')"
   
   # Check CUDA availability (optional)
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 🎯 Usage

### Running Inference

The main inference script processes the MRBench dataset with multiple language models:

```bash
cd code
python inference.py
```

**What the script does:**
1. Loads MRBench_V1.json dataset (192 samples)
2. Processes each sample with appropriate prompts (MathDial or Bridge)
3. Generates responses using Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.1
4. Saves results to `data/llama_result_MRBench_V1.json` and `data/mistral_result_MRBench_V1.json`

### Configuration Options

You can modify the following parameters in `inference.py`:

- **Models**: Add/remove models in the `model_ids` dictionary
- **Batch size**: Adjust `batch_size` based on your GPU memory
- **Generation parameters**: Modify `gen_cfg` for different generation settings
- **Dataset**: Change `dataset_file` to use MRBench_V2.json

### Data Analysis

Use the Jupyter notebook for result analysis:

```bash
jupyter notebook code/data_analysis.ipynb
```

## 📊 Dataset

### MRBench Dataset

The MRBench (Math Reasoning Benchmark) dataset contains:
- **192 samples** in MRBench_V1.json
- **Conversation histories** between tutors and students
- **Ground truth solutions** for mathematical problems
- **Data sources**: MathDial and Bridge datasets

### Sample Data Structure

```json
{
  "conversation_id": "930-b01cb51d-748d-460c-841a-08e4d5cd5cc7",
  "conversation_history": "Tutor: Hi, could you please provide...",
  "Data": "MathDial",
  "Split": "test",
  "Topic": "Not Available",
  "Ground_Truth_Solution": "Elliott took half of his steps...",
  "anno_llm_responses": {
    "Gemini": {"response": "...", "annotation": {...}},
    "Phi3": {"response": "...", "annotation": {...}}
  }
}
```

## 🤖 Models

### Supported Models

1. **Llama-3.1-8B-Instruct**
   - Parameter count: 8B
   - Optimized for instruction following
   - Memory requirement: ~16GB GPU memory

2. **Mistral-7B-Instruct-v0.1**
   - Parameter count: 7B  
   - Fast inference speed
   - Memory requirement: ~14GB GPU memory

### Model Configuration

- **Precision**: FP16 (torch.float16) for memory efficiency
- **Generation**: Greedy decoding (do_sample=False)
- **Max tokens**: 200 new tokens per response
- **Temperature**: 0.1 for focused responses

## 📈 Results

### Expected Output Files

After running inference, you'll get:

- `data/llama_result_MRBench_V1.json`: Llama model results
- `data/mistral_result_MRBench_V1.json`: Mistral model results

### Result Structure

```json
{
  "result": "Generated tutor response",
  "Data": "MathDial",
  "conversation_history": "Original conversation",
  "Topic": "Problem topic",
  "Ground_Truth_Solution": "Correct solution",
  "conversation_id": "Unique identifier"
}
```

### Evaluation Metrics

The helper functions support evaluation across multiple dimensions:
- Mistake identification
- Mistake location accuracy  
- Answer revelation
- Guidance provision
- Response coherence
- Actionability
- Tutor tone
- Human-likeness

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in inference.py
   batch_size = 8  # or smaller
   ```

2. **Model Download Issues**
   ```bash
   # Ensure you have HuggingFace access
   huggingface-cli login
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

4. **Path Issues**
   ```bash
   # Ensure you're in the code/ directory
   cd code
   python inference.py
   ```

### Performance Optimization

- **GPU Memory**: Use smaller batch sizes if you encounter memory issues
- **CPU Inference**: Remove `device_map="auto"` to use CPU (much slower)
- **Mixed Precision**: Already enabled with `torch_dtype=torch.float16`

## 📝 Citation

If you use this code or dataset in your research, please cite the original paper:

```bibtex
@inproceedings{author2025unifying,
  title={Unifying AI Tutor Evaluation},
  author={Author Name et al.},
  booktitle={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

## 🤝 Contributing

This is a research replication project. For issues or improvements:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the original paper for methodology details
3. Open an issue with detailed error messages and system information

## 📄 License

This project is for academic research purposes. Please refer to the original paper and dataset licenses for usage rights.

---

**Note**: This replication aims to faithfully reproduce the original paper's methodology. For the most up-to-date information, please refer to the original publication.
