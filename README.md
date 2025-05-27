# Building-a-GPT-2-Transformer-Based-Model-from-Scratch_PR
# GPT-2 from Scratch: Text Generation with TinyStories

This repository contains a from-scratch implementation of a GPT-2-like transformer model in PyTorch, designed for text generation and trained on the TinyStories dataset. The model features positional encoding, multi-head self-attention, and transformer decoder layers, achieving coherent story generation with a test perplexity of approximately 25.4.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Code](#running-the-code)
  - [Local Machine](#local-machine)
  - [Google Colab](#google-colab)
- [Trained Model Weights](#trained-model-weights)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Project Overview
This project implements a scaled-down GPT-2 model to generate short, coherent stories. Key features include:
- **Model Architecture**: 6 transformer decoder layers, 8 attention heads, 512-dimensional embeddings, and a feed-forward dimension of 2048.
- **Dataset**: TinyStories, a collection of simple stories, split into 70% training (189,000 samples), 20% validation (54,000), and 10% test (27,000).
- **Training**: Uses AdamW optimizer, mixed-precision training, gradient accumulation, and early stopping.
- **Evaluation**: Measures perplexity and generates sample texts.
- **Outputs**: Training metrics, plots, and generated texts saved as JSON and PNG files.

## Prerequisites
- **Python**: 3.8 or higher
- **Libraries**:
  - PyTorch 2.0+
  - Transformers (Hugging Face)
  - Datasets (Hugging Face)
  - Matplotlib
  - Tqdm
- **Hardware**: CUDA-compatible GPU recommended for faster training; CPU supported but slower.

Install dependencies:
```bash
pip install torch transformers datasets matplotlib tqdm
```

## Project Structure
- **`pattern_project.py`**: Core script with model implementation, dataset preprocessing, training, evaluation, and text generation.
- **`gpt2_best_model.pt`**: Trained model weights (generated after training; hosted externally if large).
- **`training_curves_with_accuracy.png`**: Plots of training/validation loss, perplexity, and accuracy.
- **`training_history.json`**: Training metrics and sample generated texts.
- **`report.pdf`**: Project report (generated separately, e.g., via a provided script).

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Run the pip command above to install required libraries.

## Running the Code
### Local Machine
1. **Execute the Main Script**:
   ```bash
   python pattern_project.py
   ```
   This:
   - Downloads and preprocesses the TinyStories dataset.
   - Trains the model for up to 10 epochs with early stopping (patience=3).
   - Saves the best model weights as `gpt2_best_model.pt`.
   - Generates sample texts for prompts like "Once upon a time".
   - Saves metrics in `training_history.json` and plots in `training_curves_with_accuracy.png`.

2. **Generate Text with Trained Model**:
   ```python
   from pattern_project import GPT2Model, generate_text
   from transformers import GPT2Tokenizer
   import torch

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   tokenizer.pad_token = tokenizer.eos_token
   model = GPT2Model(vocab_size=50257, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=256)
   model.load_state_dict(torch.load('gpt2_best_model.pt')['model_state_dict'])
   model.to(device)
   text = generate_text(model, tokenizer, prompt="Once upon a time", max_length=100, device=device)
   print(text)
   ```

### Google Colab
1. **Open Colab**:
   Create a new notebook at [Google Colab](https://colab.research.google.com).

2. **Enable GPU**:
   Go to `Runtime > Change runtime type > GPU` for faster training.

3. **Install Dependencies**:
   ```bash
   !pip install torch transformers datasets matplotlib tqdm
   ```

4. **Upload Script**:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload pattern_project.py
   ```

5. **Run Script**:
   ```bash
   !python pattern_project.py
   ```

6. **Download Outputs**:
   ```python
   from google.colab import files
   files.download('gpt2_best_model.pt')
   files.download('training_curves_with_accuracy.png')
   files.download('training_history.json')
   ```

7. **Generate Text**:
   Use the same code as in the local setup (step 2 above) in a Colab cell.

## Trained Model Weights
- **File**: `gpt2_best_model.pt` (contains model state dictionary and configuration).
- **Generation**: Created during training by `pattern_project.py`.
- **Usage**: Place `gpt2_best_model.pt` in the project root and use the text generation code above.
- **Regeneration**: If weights are unavailable, run `pattern_project.py` to recreate them (GPU recommended).

## Expected Results
- **Test Perplexity**: ~25.4
- **Sample Generated Text**:
  - **Prompt**: "Once upon a time"
  - **Generated**: "Once upon a time, there was a little boy named Tim who loved to explore the forest. One day, he found a shiny stone that glowed brightly..."
  - **Prompt**: "In a magical forest"
  - **Generated**: "In a magical forest, a rabbit named Bella hopped through glowing flowers. She found a hidden cave with sparkling crystals..."
- **Outputs**:
  - `training_curves_with_accuracy.png`: Visualizes training/validation loss, perplexity, and accuracy.
  - `training_history.json`: Contains metrics and generated texts.

## Troubleshooting
- **CUDA Out-of-Memory**: Reduce `batch_size` in `config` (e.g., to 2) or increase `accum_steps` in `train_model`.
- **Dataset Download Issues**: Ensure internet connectivity for Hugging Face’s `datasets` library to access TinyStories.
- **Tokenizer Errors**: Verify the GPT-2 tokenizer downloads correctly with `GPT2Tokenizer.from_pretrained('gpt2')`.
- **Slow Training**: Use a GPU (local or Colab) for faster training; CPU training is significantly slower.




## Contact
For issues, open a GitHub issue.
