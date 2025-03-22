from typing import Optional
from llm import GPTModel, generate_text, create_dataloader_v1
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import tiktoken

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def text_to_ids(text: str, tokenizer) -> Tensor:
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def ids_to_text(token_ids: Tensor, tokenizer) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device) -> Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: DataLoader, model, device, num_batches: Optional[int] = None) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        return total_loss / num_batches


def train_model(model, train_loader: DataLoader, test_loader: DataLoader, optimizer,
                device, num_epochs: int, eval_freq, eval_iter: int, start_context,
                tokenizer):
    pass


def main():
    start_context = 'Every effort moves you'
    torch.manual_seed(30)
    model = GPTModel(GPT_CONFIG)
    model.eval
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = generate_text(model,
                               idx=text_to_ids(start_context, tokenizer),
                               max_new_tokens=10,
                               context_size=GPT_CONFIG["context_length"])
    print(f'output: {ids_to_text(token_ids, tokenizer)}')
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ratio = 0.9
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    test_data = raw_text[split_idx:]
    train_loader = create_dataloader_v1(train_data, batch_size=2,
                                        max_length=GPT_CONFIG["context_length"],
                                        stride=GPT_CONFIG["context_length"],
                                        drop_last=True,
                                        shuffle=True,
                                        num_workers=0)
    test_loader = create_dataloader_v1(test_data, batch_size=2,
                                        max_length=GPT_CONFIG["context_length"],
                                        stride=GPT_CONFIG["context_length"],
                                        drop_last=False,
                                        shuffle=False,
                                        num_workers=0)
    device = 'cpu'
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(test_loader, model, device)
    print(f'training loss: {train_loss:.3f}\nvalidation loss: {val_loss:.3f}')

if __name__ == '__main__':
    main()