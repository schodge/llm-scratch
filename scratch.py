import tiktoken
import torch
from llm import GPTModel, GPT_CONFIG_124M
tokenizer = tiktoken.get_encoding('gpt2')
batch: list[torch.Tensor] = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

print(batch)

torch.manual_seed(30)
model = GPTModel(GPT_CONFIG_124M)
logits = model.forward(batch)
print(f'Output shape: {logits.shape}')
print(logits)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
print(f'Token embedding layer shape: {model.tok_emb.weight.shape}')
print(f'Output layer shape: {model.out_head.weight.shape}')
total_params_gpt2 = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)
print(f'Num trainable params: {total_params_gpt2:,}')
size_mb = total_params * 4 / (1024**2)
print(f'total size mb: {size_mb}')