import re

def clean_text(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r"[^A-Za-z\s\n]", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    text = text.lower()

    return text


def build_vocab(text):

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return chars, vocab_size, stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(tokens, itos):
    return ''.join([itos[i] for i in tokens])