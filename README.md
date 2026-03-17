# GPT From Scratch — Educational Implementation

This repository contains my implementation of a small GPT-style language model built from scratch using PyTorch.

The purpose of this project is to **understand how modern language models work internally** by implementing each component step by step instead of relying on high-level libraries.

The implementation starts from a very simple language model (Bigram) and gradually builds toward a **Transformer-based GPT architecture**.

This project is inspired by the tutorial by Andrej Karpathy:
"Let's build GPT: from scratch, in code, spelled out".

However, the goal here is **learning and reimplementation**, not copying the original code.

---

# What is a Language Model?

A language model is a neural network that learns to **predict the next token in a sequence**.

For example:

Input text

```
the cat sat on the
```

The model predicts the most likely next token

```
mat
```

By repeating this prediction step many times, the model can generate entire sentences and paragraphs.

This is the fundamental principle behind many modern AI systems such as chatbots and text generators.

---

# How This AI Works

The model in this repository is an **autoregressive transformer language model**.

It works through the following steps:

### 1. Tokenization

Text cannot be processed directly by neural networks.

The text is first converted into tokens.

Example:

```
hello
```

Character tokens:

```
[h, e, l, l, o]
```

Each token is mapped to an integer index.

---

### 2. Embedding

Each token index is converted into a **vector representation**.

Example

```
token -> vector
h -> [0.12, -0.7, 0.45, ...]
```

These vectors allow the neural network to work with numerical data.

---

### 3. Self Attention

Self-attention allows the model to look at **other tokens in the sentence** when making predictions.

Example:

```
the animal didn't cross the street because it was tired
```

The word **"it"** should refer to **animal**, not street.

Self-attention helps the model learn such relationships.

---

### 4. Transformer Blocks

A transformer block consists of:

• Self Attention
• Feed Forward Network
• Residual Connections
• Layer Normalization

Multiple transformer blocks stacked together allow the model to learn complex patterns in language.

---

### 5. Training Objective

The model is trained to **predict the next token in the sequence**.

Example training sample:

```
input : hello worl
target: ello world
```

The model gradually adjusts its parameters to minimize prediction error.

---

### 6. Text Generation

Once trained, the model generates text by:

1. Starting with a prompt
2. Predicting the next token
3. Appending it to the sequence
4. Repeating the process

This creates new text based on patterns learned during training.

---

# Dataset

The dataset used in this repository is **different from the one used in the original tutorial**.

The text was cleaned and modified before training.

Preprocessing steps include:

• removing special characters
• normalizing text
• preparing it for character level tokenization

This allows experimentation with training behavior on a custom dataset.

---

# Training

Training the model consists of repeatedly predicting the next token and updating the network parameters using gradient descent.

Typical steps:

```
1. Load dataset
2. Convert text into tokens
3. Create training batches
4. Forward pass through the model
5. Compute loss
6. Backpropagation
7. Update weights
```

---

# Why Build This From Scratch?

Implementing a model from scratch helps understand:

• how transformers work
• how tokenization works
• how attention mechanisms function
• how language models generate text

Instead of using large frameworks directly, this project focuses on learning the **core ideas behind modern AI systems**.

---

# Acknowledgment

This project is inspired by the educational work of Andrej Karpathy and his tutorial:

"Let's build GPT: from scratch, in code, spelled out".

This repository represents my own implementation created while learning these concepts.

---

# Future Improvements

Planned improvements include:

• larger datasets
• better tokenizer
• larger transformer architecture
• improved training pipeline
• experimentation with different model sizes

---

# License

MIT
