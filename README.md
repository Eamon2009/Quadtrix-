# GPT From Scratch (Learning Implementation)

This repository contains my implementation of a small GPT-style language model built from scratch using PyTorch.
The goal of this project is to understand how transformer-based language models work internally by implementing each component step by step.

This project focuses on learning concepts such as tokenization, embeddings, attention mechanisms, and training a basic autoregressive language model.

---

## Project Goal

The main objective of this repository is **educational**.
Instead of using high-level libraries, the model is implemented manually to understand the mechanics behind modern language models.

Key components implemented:

* Character-level tokenizer
* Bigram language model
* Transformer-based GPT model
* Self-attention mechanism
* Training loop
* Text generation

---

## Project Structure

```
repo/
│
├── train.py              # Transformer-based GPT model
├── bigram.py           # Simple bigram language model baseline
├── notebooks/
│   └── gpt-from-scratch.ipynb   # Notebook used for experimentation and explanations
├── cleaned.txt
├── traindata.txt
│
└── README.md
```

---

## Dataset

The dataset used in this repository is **different from the one used in the original tutorial**.

The text was **cleaned and modified** before training.
Preprocessing steps include:

* removing special characters
* normalizing the text
* preparing it for character-level tokenization

This modification was done to experiment with training behavior on a different dataset.

---

## Acknowledgment

This project is inspired by the tutorial by Andrej Karpathy:

"Let's build GPT: from scratch, in code, spelled out"

The purpose of this repository is to **reimplement and understand the concepts explained in the tutorial**, while writing the code independently as part of my learning process.

---

## Future Improvements

Planned improvements include:

* better tokenizer
* larger dataset
* improved training stability
* experimenting with larger transformer models

---

## License

MIT 
