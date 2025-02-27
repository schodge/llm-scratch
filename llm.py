import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

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

    def encode(self, text: list[str]) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    def decode(self, ids: list[int]) -> list[str]:
        return self.tokenizer.decode(ids)



class GPTDatasetV1(Dataset):
    def __init__(self, txt: list[str], tokenizer: SimpleTokenizerV3, max_length: int, stride: int) -> None:
        self.input_ids: list[int] = []
        self.target_ids: list[int] = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return (self.input_ids[idx], self.target_ids[idx])


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


def main():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=1,
                                       max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)



if __name__ == '__main__':
    main()