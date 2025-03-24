import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def evaluate_model(model: GPTModel, train_loader: DataLoader,
                   val_loader: DataLoader, device, eval_iter: int) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: GPTModel, tokenizer, device, start_context) -> None:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
        decoded_text = ids_to_text(token_ids, tokenizer)
    print(f'\t{decoded_text.replace('\n', ' ')}')
    model.train()


def train_model(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, optimizer,
                device, num_epochs: int, eval_freq: int, eval_iter: int, start_context,
                tokenizer) -> tuple[list[float], list[float], list[int]]:
    train_losses: list[float] = []
    val_losses: list[float] = []
    track_tokens_seen: list[int] = []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'epoch {epoch + 1} - step {global_step:06d}\ttrain loss: {train_loss:.3f}\tval loss: {val_loss:.3f}')
                generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label='training loss')
    ax1.plot(epochs_seen, val_losses, linestyle='-.', label='validation loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_label('tokens seen')
    fig.tight_layout()
    plt.show()


def main():
    start_context = 'Every effort moves you'
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG)
    model.eval()
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ratio = 0.9
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    test_data = raw_text[split_idx:]
    train_loader = create_dataloader_v1(train_data,
                                         batch_size=2,
                                        max_length=GPT_CONFIG["context_length"],
                                        stride=GPT_CONFIG["context_length"],
                                        drop_last=True,
                                        shuffle=True,
                                        num_workers=0)
    test_loader = create_dataloader_v1(test_data,
                                        batch_size=2,
                                        max_length=GPT_CONFIG["context_length"],
                                        stride=GPT_CONFIG["context_length"],
                                        drop_last=False,
                                        shuffle=False,
                                        num_workers=0)
    device = 'cpu'
    model.to(device)
    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_loader, model, device)
    #     val_loss = calc_loss_loader(test_loader, model, device)
    # print(f'training loss: {train_loss:.3f}\nvalidation loss: {val_loss:.3f}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer,
        device=device, num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context=start_context,
        tokenizer=tokenizer
    )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

if __name__ == '__main__':
    main()