import re
import tiktoken

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



def main():
    with open('the-verdict.text', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    tokenizer = SimpleTokenizerV3()
    enc_text = tokenizer.encode(raw_text)
    


if __init__ == '__main__':
    main()