# Unifying AI Tutor Evaluation Paper Replication

Replication of: [Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors](https://arxiv.org/pdf/2412.09416)  
*Rose E. Wang, Pawan Wirawarn, Noah Goodman, Dorottya Demszky (2024). arXiv preprint arXiv:2412.09416*

## 📊 Dataset Description

**MRBench Dataset**: A comprehensive benchmark for evaluating AI tutors' pedagogical abilities across multiple dimensions.

| File | Description |
|------|-------------|
| `MRBench_V1.json` | Original dataset containing 192 dialogues as detailed in the paper |
| `MRBench_V2.json` | Updated version with additional 8 dialogues, bringing the total to 200 examples |

*Dataset Describtion*

1. `conversation_id`: Serves as a unique identifier to track each dialogue in the dataset.
2. `conversation_history`: Captures the dialogue context relevant to the ongoing interaction.
3. `Data`: Specifies the dataset used for the interaction, such as MathDial or Bridge.
4. `Split`: Indicates whether the data point belongs to the test, train, or validation set.
5. `Topic`: Categorizes the dialogue into broad sub topics in Mathematics for easier filtering and analysis.
6. `Ground_Truth_Solution`: Provides a step-by-step solution to the problem discussed in the conversation, serving as a gold standard for evaluation.
7. `anno_llm_responses`: Stores LLM-specific responses with detailed annotations for evaluation based on multiple dimensions like mistake identification, guidance, etc.


## 🚀 Quick Start

#### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run Model Inference
```bash
# Generate responses using Llama models
python code/inference_llama.py

# Generate responses using Mistral models  
python code/inference_mistralai.py

# Evaluate model responses using llama
python code/inference_llama_eval.py
```


### 3. Data Analysis
Open and run the Jupyter notebooks for detailed analysis(Metric):
```bash
jupyter notebook code/data_analysis.ipynb
```

## 🔬 Experimental Results

### Table 3: Pedagogical Ability Assessment Results(Origin Data Analysis, W/o inference)

Performance comparison between original paper and our replication across 8 evaluation dimensions:

**Difference = Our Results - Paper Results**

| Tutor | Mistake_Identification | Mistake_Location | Revealing_of_the_Answer | Providing_Guidance | Actionability | Coherence | Tutor_Tone | Human-likeness |
|-------|:---------------------:|:----------------:|:----------------------:|:------------------:|:-------------:|:---------:|:----------:|:--------------:|
| **Expert** | | | | | | | | |
| Paper | 76.04 | 63.02 | 90.62 | 67.19 | 76.04 | 79.17 | 92.19 | 87.50 |
| Our | 81.25 | 68.75 | 97.92 | 72.92 | 81.77 | 84.90 | 17.19 | 94.79 |
| Diff | **+5.21** | **+5.73** | **+7.30** | **+5.73** | **+5.73** | **+5.73** | **-75.00** | **+7.29** |
| **GPT-4** | | | | | | | | |
| Paper | 94.27 | 84.38 | 53.12 | 76.04 | 46.35 | 90.17 | 37.50 | 89.62 |
| Our | 94.27 | 85.42 | 54.69 | 77.08 | 46.88 | 92.71 | 36.98 | 93.23 |
| Diff | **0.00** | **-1.04** | **-1.57** | **-1.04** | **-0.53** | **-2.54** | **+0.52** | **-3.61** |
| **Gemini** | | | | | | | | |
| Paper | 63.02 | 39.58 | 67.71 | 37.50 | 42.71 | 56.77 | 21.88 | 68.23 |
| Our | 87.50 | 62.50 | 92.71 | 58.85 | 61.98 | 82.29 | 39.58 | 95.31 |
| Diff | **+24.48** | **+22.92** | **+25.00** | **+21.35** | **+19.27** | **+25.52** | **+17.70** | **+27.08** |
| **Llama3.1-405B** | | | | | | | | |
| Paper | 94.27 | 84.38 | 80.73 | 77.08 | 74.48 | 91.67 | 16.15 | 90.62 |
| Our | 95.31 | 84.90 | 81.77 | 77.60 | 75.52 | 94.27 | 17.71 | 93.23 |
| Diff | **+1.04** | **+0.52** | **+1.04** | **+0.52** | **+1.04** | **+2.60** | **+1.56** | **+2.61** |
| **Llama3.1-8B** | | | | | | | | |
| Paper | 80.21 | 54.69 | 73.96 | 45.31 | 42.71 | 80.73 | 19.79 | 93.75 |
| Our | 81.25 | 56.25 | 76.56 | 46.88 | 42.71 | 82.81 | 19.79 | 96.35 |
| Diff | **+1.04** | **+1.56** | **+2.60** | **+1.57** | **0.00** | **+2.08** | **0.00** | **+2.60** |
| **Mistral** | | | | | | | | |
| Paper | 93.23 | 73.44 | 86.46 | 63.54 | 70.31 | 86.98 | 15.10 | 95.31 |
| Our | 93.23 | 74.48 | 89.06 | 66.15 | 71.35 | 88.02 | 16.67 | 97.40 |
| Diff | **0.00** | **+1.04** | **+2.60** | **+2.61** | **+1.04** | **+1.04** | **+1.57** | **+2.09** |
| **Sonnet** | | | | | | | | |
| Paper | 85.42 | 69.79 | 94.79 | 59.38 | 60.94 | 88.54 | 54.69 | 96.30 |
| Our | 86.98 | 71.35 | 96.88 | 63.02 | 62.50 | 90.62 | 57.81 | 98.96 |
| Diff | **+1.56** | **+1.56** | **+2.09** | **+3.64** | **+1.56** | **+2.08** | **+3.12** | **+2.66** |

#### Key Findings

- **Overall Performance**: Our replication achieves comparable or better performance across most tasks and models
- **Significant Improvements**: 
  - Gemini shows the largest improvements across all dimensions (+17% to +27%)
  - Expert human tutors show consistent improvements (+5% to +7%) except for Tutor_Tone
- **Consistent Results**: Most models show small positive improvements, indicating successful replication
- **Notable Discrepancy**: Expert Tutor_Tone shows a large negative difference (-75%), which may indicate different annotation criteria or data preprocessing


## TODO

1. replicate Table 3 (LLama & Mistral at first)
2. replicate Table 5 and Tabele 6 results

