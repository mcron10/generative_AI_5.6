# generative_AI_5.6
## Udacity Generative AI 5.6
## Parameter-Efficient Fine-Tuning (PEFT) using LoRA on IMDb Sentiment Analysis

This project demonstrates how to fine-tune a Hugging Face pre-trained Transformer model using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA). Specifically, it adapts the DistilBERT model to classify movie reviews from the IMDb dataset into positive and negative sentiment categories.

# Overview

The goal of this project is to illustrate how lightweight fine-tuning techniques like LoRA can significantly enhance model performance with minimal computational resources.

# Project Steps

## 1. Loading and Evaluating the Foundation Model

Load the IMDb dataset (subset of 500 samples each for training and testing).

Initialize a pre-trained DistilBERT model (distilbert-base-uncased) for sequence classification.

Evaluate initial performance to establish a baseline.

## 2. Performing Parameter-Efficient Fine-Tuning

Configure and apply LoRA to adapt the pre-trained model efficiently.

Train the PEFT model using the configured LoRA settings for one epoch.

Save the fine-tuned adapter weights.

## 3. Performing Inference with the Fine-Tuned PEFT Model

Load the saved PEFT model.

Evaluate the fine-tuned model performance and compare it against the pre-trained baseline.

# Key Results

Pre-trained Model Accuracy: ~5%

Fine-tuned PEFT Model Accuracy: 100%

This demonstrates a significant improvement due to the effectiveness of parameter-efficient fine-tuning.

# Setup

## Requirements

Python 3.8+

Transformers

Datasets

PEFT

PyTorch

scikit-learn

Install dependencies:

pip install transformers datasets peft torch scikit-learn

# How to Run

Execute the provided Jupyter notebook (.ipynb) or Python script (.py) sequentially to replicate the results:

python your_script_name.py

# Directory Structure

project/
├── lora_imdb_model/         # Directory containing saved PEFT fine-tuned weights
├── lora_imdb_results/       # Output directory for training checkpoints
├── your_script_name.py      # Project code
└── README.md                # This README

# References

Hugging Face Transformers

Hugging Face PEFT Library

IMDb Dataset
