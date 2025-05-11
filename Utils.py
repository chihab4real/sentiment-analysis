import torch
from torch.utils.data import Dataset
import re
import pandas as pd

class MovieReviewDataset(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_length):
        self.reviews = reviews.tolist()
        self.sentiments = sentiments.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.reviews[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.sentiments[idx], dtype=torch.long)
        }




def clean_text(text):
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = text.lower()
   
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    
    return pd.read_csv(file_path)


def clean_df(df, columns):
    for column in columns:
        df[column] = df[column].apply(clean_text)
    return df

def map_labels(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df
