import argparse
import pandas as pd
import tiktoken
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torch
from torch import Tensor

RANDOM_SEED = 30

class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: Optional[int] = None, pad_token_id: int = 50256) -> None:
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index: int) -> Tensor:
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _longest_encoded_length(self) -> int:
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    num_spam = df[df['Label'] == 'spam'].shape[0]
    ham_sub = df[df['Label'] == 'ham'].sample(num_spam, random_state=RANDOM_SEED)
    balanced_df = pd.concat([ham_sub, df[df['Label'] == 'spam']])
    return balanced_df


def random_split(df: pd.DataFrame, train: float, validate: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_end = int(len(df) * train)
    valid_end = train_end + int(len(df) * validate)
    train_df = df[:train_end]
    validate_df = df[train_end:valid_end]
    test_df = df[valid_end:]
    return train_df, validate_df, test_df



def main():
    num_workers = 0
    batch_size = 8
    torch.manual_seed(RANDOM_SEED)
    tokenizer = tiktoken.get_encoding("gpt2")

    parser = argparse.ArgumentParser(description="Finetune a GPT model for classification")
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes.")
    )
    args = parser.parse_args()

    df = pd.read_csv('./sms_spam_collection/spam.tsv', sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    train_dataset = SpamDataset( csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset( csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset( csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    train_loader = DataLoader( dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader( dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader( dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)





if __name__ == '__main__':
    main()