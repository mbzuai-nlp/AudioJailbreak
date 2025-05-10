# Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/MBZUAI/AudioJailbreak)

AudioJailbreak is a benchmark framework specifically designed for evaluating the security of Audio Language Models (Audio LLMs). This project tests model defenses against malicious requests through various audio perturbation techniques.  
**Note**: This project aims to improve the security of audio language models. Researchers should use this tool responsibly.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation Guide](#installation-guide)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## 📝 Project Overview

AudioJailbreak provides a comprehensive evaluation framework for testing the robustness of audio language models against adversarial attacks. Our method incorporates carefully designed perturbations in audio inputs to test model security mechanisms. Key features include:

- **Diverse test cases**: Covering multiple categories of harmful speech samples
- **Automated evaluation pipeline**: End-to-end automation from audio processing to result analysis
- **Bayesian optimization**: Intelligent search for optimal perturbation parameters
- **Multi-model compatibility**: Support for evaluating mainstream audio language models

## 🔧 Installation Guide

1. Clone repository:
```bash
git clone https://github.com/mbzuai-nlp/AudioJailbreak
cd AudioJailbreak
```

2. Create and activate environment:
```bash
conda env create -f environment.yaml
conda activate Audiojailbreak
```

3. Download dataset (from Hugging Face):
```
Link: https://huggingface.co/datasets/MBZUAI/AudioJailbreak
```

## 💾 Dataset

**Important Notice**: This repository contains code only. All audio data and preprocessed/inference result JSONL files are hosted on [Hugging Face](https://huggingface.co/datasets/MBZUAI/AudioJailbreak).

Dataset includes:
- Original speech samples (`audio/`)
- Input JSONL files (`convert/question`)
- Model responses (`inference/response`)
- Evaluation results (`eval/xx`), where xx is model name
- Original texts (`text/`)

## 📁 Code Structure

```
AudioJailbreak/
├── audio/            # Audio processing tools (actual audio files on Hugging Face)
├── convert/          # Data conversion & formatting (actual JSONL files on Hugging Face)
├── eval/             # Evaluation scripts & metrics
├── inference/        # Model inference & adversarial testing
├── script/           # Speech conversion scripts
├── experiment/       # Threshold experiment configurations
└── text/             # Text processing utilities
```

### Core Components

- **new_add_noise.py**: Implements multiple audio perturbation methods
- **APT_BO_diva.py**: Bayesian optimization-based adversarial sample generation
- **xx.py**: Model inference interfaces (xx represents model name)
- **eval.py**: Evaluation metric calculations

## 🚀 Usage

### 1. Data Preparation

```
Link: https://huggingface.co/datasets/MBZUAI/AudioJailbreak
```

### 2. Basic Audio Jailbreaking

```bash
# Generate adversarial samples with Bayesian optimization
python infer/xx.py  # Replace xx with model name
```

### 3. APT-based Bayesian Audio Jailbreaking

```bash
# Generate adversarial samples with Bayesian optimization
python infer/APT_BO_diva.py
```

### 4. Model Evaluation

```bash
# Single model evaluation
python eval_utils/eval.py

# Batch evaluation
bash eval_utils/auto.sh
```

### 5. Result Analysis

```bash
# Generate evaluation report
python eval/all_average.py
```

## 📊 Evaluation Metrics

Our benchmark uses multiple metrics to assess model security:

- **dict_dsr**: Model refusal rate (dict_dsr=1-asr)
- **Semantic Similarity**: Semantic similarity of refusal responses
- **GPT Score**: Includes sub-metrics:
  - Harmfulness: Degree of harmful content
  - Policy Violation: Policy violation rate
  - Relevance: Response relevance

## 📄 Citation

If you use AudioJailbreak in your research, please cite our paper:



## 📜 License

This project is licensed under [Apache License 2.0](LICENSE).

---

**Note**: This project aims to improve the security of audio language models. Researchers must use this tool responsibly.
