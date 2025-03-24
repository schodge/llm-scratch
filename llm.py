from typing import Any
import re
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': False
}


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        prep = re.split(r'([,.?_!"()\']|--|\s)', text)
        cleaned = [item.strip() for item in prep if item.strip()]
        final = [item if item in self.str_to_int else "<|unk|>" for item in cleaned]
        ids = [self.str_to_int(x) for x in cleaned]
        return ids
    
    def decode(self, ids: list[int]) -> list[str]:
        text = ' '.join([self.int_to_str(x) for x in ids])
        cleaned_text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV3:
    def __init__(self) -> None:
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def encode(self, text: list[str], allowed_special={"<|endoftext|>"}) -> list[int]:
        return self.tokenizer.encode(text, allowed_special=allowed_special)
    
    def decode(self, ids: list[int]) -> list[str]:
        return self.tokenizer.decode(ids)



class GPTDatasetV1(Dataset):
    def __init__(self, txt: list[str], tokenizer: SimpleTokenizerV3, max_length: int, stride: int) -> None:
        self.input_ids: list[int] = []
        self.target_ids: list[int] = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return (self.input_ids[idx], self.target_ids[idx])




class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert (d_out % num_heads == 0), "d_out % num_heads must == 0"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec: Tensor = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


def create_dataloader_v1(txt: list[str], batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0) -> DataLoader:
    dataset = GPTDatasetV1(txt, SimpleTokenizerV3(), max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader





class LayerNorm(nn.Module):
    def __init__(self, emb_dim: tuple[int, int]) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.bias


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        temp = 0.5 * x * (
            1 + 
            torch.tanh(torch.sqrt(torch.tensor(2. / torch.pi)) *
                       (x + 0.044715 * torch.pow(x, 3))))
        return temp


class FeedForward(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
            #nn.Dropout()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.att = MultiHeadAttentionWrapper(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias'],
            dropout=cfg['drop_rate']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx: list[Tensor]) -> Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
def generate_text(model, idx: Tensor, max_new_tokens: int, context_size: int) -> Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    assert batch == Tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])

    torch.manual_seed(30)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    params = sum(p.numel() for p in model.parameters())
    tp = params - sum(p.numel() for p in model.out_head.parameters())
    print(f"output shape: {logits.shape}\n{logits}")
    print(f"total params: {params}")
    print(f"total trainable params: {tp}")

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_text(model, idx=encoded_tensor, max_new_tokens=6,
                        context_size=GPT_CONFIG_124M['context_length'])

    print(f"output length: {len(out[0])}\noutput:{out}")
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ == '__main__':
    main()