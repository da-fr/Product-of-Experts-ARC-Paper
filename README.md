<h1 align="center">
    Product of Experts with LLMs:<br/>Boosting Performance on ARC Is a Matter of Perspective
</h1>

<p align="center">
  <a href="https://da-fr.github.io/Product-of-Experts-ARC-Paper">
    <img src="https://img.shields.io/badge/TLDR-Blog-%238D6748?style=for-the-badge&logo=Website&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2505.07859">
    <img src="https://img.shields.io/badge/arXiv-2505.07859-b31b1b.svg?logo=arxiv&style=for-the-badge">
  </a>
  <a href="https://openreview.net/forum?id=dsBjxI6l8W">
    <img src="https://img.shields.io/badge/OpenReview-Discussion-2e77a5.svg?logo=openreview&style=for-the-badge">
  </a>
  <br/>
  <a href="https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit">
    <img src="https://img.shields.io/badge/🤗-Model-yellow.svg?style=for-the-badge">
  </a>
  <a href="https://www.kaggle.com/competitions/arc-prize-2024/leaderboard">
    <img src="https://img.shields.io/badge/ARC_Kaggle_Competition_2024-Leaderboard-20BEFF.svg?logo=kaggle&style=for-the-badge">
  </a>
  <a href="https://github.com/da-fr/Product-of-Experts-ARC-Paper/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge">
  </a>
</p>


Code for the ICML 2025 Paper **Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective**, providing state-of-the-art methods leveraging Large Language Models (LLMs) and algorithmic sampling strategies to achieve a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set. Includes reproducible experiments, solution evaluation tools, and detailed instructions for running the experiments presented in the paper efficiently on accessible hardware.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ec06c9ff-3bd5-418d-8d1e-0a30914da717" width="100%" height="auto" />
</p>

## Models

Both our models from the paper are available for download from Huggingface:
 * [NeMo-Minitron-8B model](https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit) (71.6% on the public ARC-AGI evaluation set)
 * [Llama-3.2-3B model](https://huggingface.co/da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit) (61.4% on the public ARC-AGI evaluation set)

## Evaluation (with test-time-training)

The primary entry points for evaluating a model on an ARC dataset are named in the format:
`evaluation_[model]_on_[dataset]_[sampling_strategy]_with_ttt.py`.

These scripts automatically download the model from Huggingface, and then process each task in the dataset sequentially, performing test-time-training followed by candidate generation and scoring. Model outputs are cached using `diskcache`, allowing quick re-runs with identical settings.

Our evaluation code requires the `unsloth` and `diskcache` packages to be installed.

Running the Sudoku evaluation additionally requires downloading the [Sudoku-3m dataset](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) and the `pandas` package to be installed.

## Initial finetuning

To re-run the initial finetuning of our [NeMo-Minitron-8B model](https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit), execute the `finetune_NemoReArc1200.py` script. The training process requires 20,000 examples per task from the ReArc dataset, which must be generated in advance using [Michael Hodel's ReArc code](https://github.com/michaelhodel/re-arc) and placed under `input/re_arc`. The training code for our weaker [Llama-3.2-3B model](https://huggingface.co/da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit) can be found in our older [Kaggle Arc Prize 2024 github repository](https://github.com/da-fr/arc-prize-2024).

Please ensure the `unsloth` package is installed before running our training code. All our models were initially finetuned on a single `Nvidia H100` GPU. If you encounter memory problems, consider reducing batch size and/or the `max_tokens` value. Using a batch size of `2` should allow finetuning `Mistral-NeMo-Minitron-8B-Base` on GPUs with 24 GB memory.

## Files

Here is a rough overview of our files and classes:

#### `arc_loader.py`
- **Purpose**: Handles all Data formatting and loading
- **Capabilities**:
   - Class `ArcDataset` which handles all data set related tasks, e.g.:
   - Building datasets from various sources.
   - Modifying, shuffling, and augmenting examples.
   - Splitting, sorting, and filtering examples.
   - Handling dataset keys, challenges and solutions.
   - Preparing the data for tokenization.
   - Creating and verifying submissions.

#### `model_tools.py`
- **Purpose**: Contains code for loading, saving and manipulating models
- **Capabilities**: 
   - Load and Save Model and LoRA adapters
   - Shrink Tokenizer and Embedding Layers
   - Data Collator for masking the task inputs and the first output

#### `inference_tools.py`
- **Purpose**: Contains tools for inference and scoring
- **Capabilities**: 
   - Inference code, including our custom DFS
   - Score calculation

#### `selection.py`
- **Purpose**: Contains functions used to select best answer from different Candidates
- **Capabilities**:
   - Various score aggregation methods
   - Sorting candidates by their score for later submission generation
   - Class `EvalTool` for doing above tasks on-the-fly and printing results

#### `finetuning_[model].py`
- **Purpose**: Run the initial finetuning process.
- **Required packages**: `unsloth`
- **Steps**:
   - Load the base model and reduce embedding size.
   - Load and augment training data.
   - Create a lora adapter and execute training.
   - Save the trained lora adapter.
   - Merge the lora model into the base model and save as final model.

#### `evaluation_[model]_on_[dataset]_[sampling_strategy]_with_ttt.py`
- **Purpose**: Run inference
- **Required packages**: `unsloth` and `diskcache`
- **Steps**:
   - Load the finetuned model.
   - Run additional finetuning and inference for each task.
   - Write the `submission.json` and `results.pickle.bz2` file.
   - Reload and verify the submission file.

## License

Our code is available under the Apache 2.0 license. See the [LICENSE.txt](LICENSE.txt) file for more info.

