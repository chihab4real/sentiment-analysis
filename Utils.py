import torch
from torch.utils.data import Dataset
import re
import pandas as pd


# Custom Dataset class for movie reviews
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


# Clean HTML tags, non-alphabet characters, and normalize text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load a CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)


# Apply cleaning to specified columns in a DataFrame
def clean_df(df, columns):
    for column in columns:
        df[column] = df[column].apply(clean_text)
    return df


# Map text labels to numeric values using a dictionary
def map_labels(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df


# Predict sentiment (positive or negative) for a given text
def predict_sentiment(model, tokenizer, device, max_length, text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(
            encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "Positive" if prediction == 1 else "Negative"
