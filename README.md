# Replication Package: When AI Coding Assistants Leak Training Data

This replication package contains the scripts and analysis code for the paper "When AI Coding Assistants Leak Training Data: Study Memorization in Code LLMs". 

## Paper Overview

This study investigates memorization behavior in state-of-the-art code LLMs using a two-stage attack pipeline combining membership inference and data extraction. The paper evaluates four models on a custom dataset of 30,000+ Python files.

### Evaluation Dataset

The evaluation dataset used in this study was constructed from GitHub repositories using the BigQuery `github-repos` dataset (see [BigQuery Paper](https://dl.acm.org/doi/10.1145/2790755.2790797)). The dataset construction process:

1. **Source**: Over 30,000 random Python files extracted from GitHub repositories via BigQuery
2. **Processing**: Files were tokenized using each target model's default tokenizer
3. **Sampling**: Randomly sampled 150-token sequences from each file
4. **Structure**: Each sequence was split into three equal 50-token parts:
   - **Pre-prefix** (50 tokens): Context before the target code
   - **Prefix** (50 tokens): Prompt for code completion
   - **Suffix** (50 tokens): Ground truth target code for comparison

This diverse corpus minimizes evaluation bias and provides a representative sample of real-world Python code patterns.

*The experiment results of this paper is avaialble separately [https://figshare.com/s/542e6c8b950966cfe2b1](https://figshare.com/s/542e6c8b950966cfe2b1)*

### Models Evaluated

1. **StarCoder2-3B** - Code-specialized, 3B parameters
2. **StarCoder2-7B** - Code-specialized, 7B parameters
3. **Llama3-8B** - General-purpose, 8B parameters
4. **DeepSeek-R1-distilled-Llama-8B** - Distilled version of Llama3-8B



## Package Structure

```
replication_package/
├── README.md (this file)
├── requirements.txt
├── exp1_mia_and_extraction/          # Section 4.1-4.2: MIA and Extraction (uses exp1 & exp2 data)
│   ├── MIA_result_analysis_and_presentation.ipynb
│   ├── plot_MIA_and_Extraction_Results.ipynb
│   └── extraction_analysis.ipynb
├── exp2_extraction/                  # Section 4.2: Data Extraction (empty - see README inside)
│   └── README.md                     # Explains why this folder is empty
├── exp3_categorization/              # Section 4.3: Categorization Analysis (uses exp3 data)
│   ├── categorization_for_llama.ipynb
│   └── plot_category_analysis.ipynb
├── exp4_attribute_analysis/          # Section 4.4: Attribute Analysis (uses exp4 data)
│   ├── Code_Attributes_Analysis.ipynb
│   ├── step1_method_extraction.ipynb
│   ├── step2_finding_preceding_context.ipynb
│   ├── step3_extraction_with_context_head_limit_output.ipynb
│   ├── step4_attribute_analysis.ipynb
│   └── plot_unintentional_attack_result_graph.ipynb
└── exp5_distillation/                # Section 4.5: Knowledge Distillation (uses exp5 data)
    └── plot_distillation_result_analysis.ipynb
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Before running any scripts that require API access, set your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

Or create a `.env` file in the replication_package directory:

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. Configure Local Models (StarCoder, Llama, etc.)

For experiments that use locally deployed models (e.g., StarCoder2, Llama3), you need to configure the following:

#### Required Dependencies

The following packages are needed for local model deployment:
- `transformers` - For loading models from HuggingFace
- `torch` - PyTorch for model inference
- `huggingface_hub` - For downloading models from HuggingFace
- `nltk` - For BLEU score calculations

Install them with:
```bash
pip install transformers torch huggingface_hub nltk
```

#### Model Access

Some models require HuggingFace authentication:
- **StarCoder2 models** (`bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`): Publicly available, no token needed
- **Llama models** (`meta-llama/Llama-3.1-8B`): May require HuggingFace token for gated access

If needed, set your HuggingFace token:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or login via command line:
```bash
huggingface-cli login
```

#### Model Configuration

In notebooks that use local models (e.g., `exp4_attribute_analysis/step3_extraction_with_context_head_limit_output.ipynb`), you'll need to configure:

1. **Device**: Set to `'cuda'` for GPU or `'cpu'` for CPU
   ```python
   device = 'cuda'  # or 'cpu'
   ```

2. **Model Checkpoint**: Specify the model name
   ```python
   checkpoint = "bigcode/starcoder2-7b"  # or "bigcode/starcoder2-3b", "meta-llama/Llama-3.1-8B"
   ```

3. **Model Loading**: Load tokenizer and model
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch
   
   tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   model = AutoModelForCausalLM.from_pretrained(
       checkpoint, 
       device_map=device,
       torch_dtype=torch.float16  # Use FP16 for memory efficiency
   )
   ```

4. **Hardware Requirements**:
   - **GPU recommended**: Models require significant memory (7B models need ~14GB VRAM for FP16)
   - **CPU fallback**: Can use CPU but will be significantly slower
   - **Memory**: Ensure sufficient RAM/VRAM for model loading


### 4. Update Data Paths

The scripts contain hardcoded paths to the experimental results. You will need to update these paths in each notebook to point to your local data directory. The paths are typically found at the beginning of each notebook.

**Example:** Replace paths like:
```python
base_path = "/Users/anonymous_user/Downloads/experiment_results/..."
```

With your local path:
```python
base_path = "/path/to/your/experiment_results/..."
```

Alternatively, you can set an environment variable:
```bash
export EXPERIMENT_RESULTS_PATH="/path/to/your/experiment_results"
```

And update the notebooks to use:
```python
import os
base_path = os.path.join(os.environ.get("EXPERIMENT_RESULTS_PATH", "."), "exp1_membership_inference_attacks_results/...")
```

## Running the Experiments

### Experiment 1: Membership Inference and Extraction (Section 4.1-4.2)

This experiment evaluates the effectiveness of membership inference attacks and data extraction on the code LLMs.

**Scripts:**
1. `MIA_result_analysis_and_presentation.ipynb` - Main analysis of MIA results
2. `plot_MIA_and_Extraction_Results.ipynb` - Generate plots for MIA and extraction results
3. `extraction_analysis.ipynb` - Detailed extraction analysis

**Expected Results:**
- Memorization rates ranging from 42% to 64% across different models
- BLEU score distributions showing exact matches (BLEU=1.0)

**To Run:**
1. Ensure experimental results data is in the correct location
2. Update file paths in the notebooks
3. Run cells sequentially

### Experiment 3: Categorization Analysis (Section 4.3)

This experiment categorizes extracted code snippets into five categories: license, documentation, dictionaries, code logic, and testing.

**Scripts:**
1. `categorization_for_llama.ipynb` - Performs categorization using DeepSeek API
2. `plot_category_analysis.ipynb` - Generates plots showing memorization rates by category

**Expected Results:**
- Categorization of code snippets into five types
- Memorization rates by category (license and documentation showing highest rates ~70%)

**To Run:**
1. Set `DEEPSEEK_API_KEY` environment variable
2. Update input/output paths in `categorization_for_llama.ipynb`
3. Run the categorization notebook first
4. Then run the plotting notebook to visualize results

**Note:** The categorization script uses the DeepSeek API and may incur API costs. Consider processing data in batches.

### Experiment 4: Attribute Analysis - Unintentional Attacks (Section 4.4)

This experiment analyzes unintentional memorization in realistic code completion scenarios.

**Scripts (run in order):**
1. `Code_Attributes_Analysis.ipynb` - Main analysis pipeline
2. `step1_method_extraction.ipynb` - Extract methods from categorized code
3. `step2_finding_preceding_context.ipynb` - Find preceding context for methods
4. `step3_extraction_with_context_head_limit_output.ipynb` - Perform extraction with context
5. `step4_attribute_analysis.ipynb` - Statistical analysis of attributes
6. `plot_unintentional_attack_result_graph.ipynb` - Generate visualization plots

**Expected Results:**
- Unintentional memorization rates of 13-14% in realistic scenarios
- Statistical correlations between code attributes and memorization
- Logistic regression models showing factors affecting memorization

**To Run:**
1. Ensure previous experiment results are available (requires Exp3 categorization results)
2. Set `DEEPSEEK_API_KEY` environment variable (for steps 1-2)
3. **For step3**: Configure local model (StarCoder or Llama) - see "Configure Local Models" section above
4. Run scripts in sequential order (step1 → step2 → step3 → step4)
5. Run the plotting notebook to visualize results

**Note**: Step 3 (`step3_extraction_with_context_head_limit_output.ipynb`) requires a locally deployed model (StarCoder2 or Llama3). This step cannot use API-based models and requires GPU resources for reasonable performance.

### Experiment 5: Knowledge Distillation (Section 4.5)

This experiment evaluates the effectiveness of knowledge distillation in reducing memorization.

**Scripts:**
1. `plot_distillation_result_analysis.ipynb` - Compare memorization rates between regular and distilled models

**Expected Results:**
- Approximately 19% reduction in extraction rates for distilled model
- Comparison plots showing distillation effectiveness

**To Run:**
1. Ensure experimental results for both regular and distilled models are available
2. Update file paths in the notebook
3. Run cells sequentially



## Troubleshooting

### Common Issues

1. **API Key Errors:**
   - Ensure `DEEPSEEK_API_KEY` is set correctly
   - Check that the API key has sufficient credits/quota

2. **File Not Found Errors:**
   - Verify that experimental results data is in the expected locations
   - Update all hardcoded paths in the notebooks

3. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Restart the Jupyter kernel after installing packages

4. **Memory Issues:**
   - Some notebooks process large datasets. Consider processing in smaller batches
   - Close other applications to free up memory
