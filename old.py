from typing import Any
from torch import Tensor


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


def main():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    max_length = 4
    vocab_size = 50257
    output_dim = 256
    dataloader = create_dataloader_v1(raw_text, batch_size=8,
                                       max_length=max_length, shuffle=False)

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings
    torch.manual_seed(30)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    context_length = batch.shape[1]
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print(context_vecs.shape)
