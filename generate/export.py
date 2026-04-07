import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTLanguageModel, device, block_size, _model_path

print("Loading weights...")
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load(_model_path, map_location=device, weights_only=True))
model.eval()

print("Exporting to TorchScript...")
example = torch.zeros((1, 1), dtype=torch.long, device=device)
traced  = torch.jit.trace(model, example)

out_path = os.path.join(os.path.dirname(_model_path), 'best_model.pt')
traced.save(out_path)
print(f"Saved TorchScript model to: {out_path}")