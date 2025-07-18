# Sentiment Analysis with BERT 🎬

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art sentiment analysis model using BERT for movie review classification. This project achieves high accuracy on the IMDB dataset through fine-tuning a pre-trained BERT model for binary sentiment classification.

## 🎯 Project Overview

This project implements a comprehensive sentiment analysis pipeline using BERT (Bidirectional Encoder Representations from Transformers) to classify movie reviews as positive or negative. The implementation includes data preprocessing, model training, evaluation, and testing components with professional-grade code quality.

### Key Features
- 🧠 **BERT-based Architecture**: Utilizes `bert-base-uncased` for robust text understanding
- 📊 **Comprehensive Evaluation**: Includes accuracy, F1-score, confusion matrix, and classification reports
- 🔧 **Modular Design**: Clean, reusable code with proper separation of concerns
- 📈 **Visualization**: Training loss curves and performance metrics visualization
- 🧪 **Testing Suite**: Unit tests and code quality checks with flake8 and mypy
- 💾 **Model Persistence**: Automated model saving with configuration tracking

## 📁 Project Structure

```
sentiment-analysis/
├── 📂 data/
│   └── imdb_dataset.csv              # Original IMDB dataset
├── 📂 cleaned_splitted_data/
│   ├── train_dataset.csv             # Training data (72% of dataset)
│   ├── val_dataset.csv               # Validation data (8% of dataset)
│   └── test_dataset.csv              # Test data (20% of dataset)
├── 📂 models/
│   ├── sentiment_analysis_model_*.pth # Trained model weights
│   ├── tokenizer/                    # BERT tokenizer files
│   └── model_config.json             # Model configuration and metrics
├── 📓 train_model.ipynb              # Model training notebook
├── 📓 test_model.ipynb               # Model testing and validation notebook
├── 🐍 Utils.py                       # Utility functions and custom dataset class
├── 📋 requirements.txt               # Project dependencies
└── 📖 README.md                      # Project documentation
```

## 👥 Team Members

- **[Chihabeddine Zitouni](https://github.com/chihab4real)** - Machine Learning Engineer
- **[Patrick Molina](https://github.com/patrickmolina1/)** - Data Scientist
- **[Małgorzata Gierdewicz](https://github.com/malgier01)** - NLP Researcher

## 📊 Dataset

**Source**: [IMDB 50K Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

- **Size**: 50,000 movie reviews
- **Balance**: 25,000 positive and 25,000 negative reviews
- **Task**: Binary sentiment classification
- **Format**: CSV with 'review' and 'sentiment' columns

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chihab4real/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Place `imdb_dataset.csv` in the `data/` directory

### Training the Model

```bash
# Open the training notebook
jupyter notebook train_model.ipynb
```

Or run all cells programmatically:
```bash
jupyter nbconvert --to script train_model.ipynb
python train_model.py
```



### Testing the Model

```bash
# Open the testing notebook
jupyter notebook test_model.ipynb
```

## 🔧 Technical Implementation

### Data Preprocessing Pipeline

```python
# Text cleaning and normalization
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    text = text.lower()                       # Lowercase normalization
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text
```

**Key Processing Steps:**
1. **HTML Tag Removal**: Strips HTML markup from review text
2. **Text Normalization**: Converts to lowercase and removes special characters
3. **Label Mapping**: `positive` → `1`, `negative` → `0`
4. **Dataset Splitting**: 72% training, 8% validation, 20% testing

### Model Architecture

```python
# BERT model configuration
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS
)
```

**Architecture Details:**
- **Base Model**: `bert-base-uncased` (110M parameters)
- **Classification Head**: Linear layer with 2 output classes
- **Tokenizer**: BERT WordPiece tokenizer with 128 max sequence length
- **Input Processing**: Attention masks and special tokens ([CLS], [SEP])

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | AdamW | Adaptive learning rate with weight decay |
| **Learning Rate** | 2e-5 | Optimal rate for BERT fine-tuning |
| **Batch Size** | 32 | Balanced for memory and convergence |
| **Epochs** | 3 | Sufficient for convergence without overfitting |
| **Max Length** | 128 | Captures most review content efficiently |

## 📈 Performance Metrics

The model achieves excellent performance on the IMDB test set:

- **Accuracy**: ~90-95% (typical range for BERT on IMDB)
- **F1 Score**: Balanced performance across both classes
- **Training Time**: ~2-3 hours on GPU, ~8-12 hours on CPU

### Evaluation Visualizations

- 📊 **Training Loss Curve**: Monitors convergence during training
- 🔥 **Confusion Matrix**: Visualizes prediction accuracy by class
- 📈 **Classification Report Heatmap**: Detailed precision/recall metrics
- 📉 **Validation Accuracy**: Tracks performance on unseen data

## 🛠️ Code Quality & Testing

### Static Analysis
```bash
# Code style checking
flake8 Utils.py

# Type checking
mypy Utils.py
```

### Unit Testing
```python
# Test data preprocessing functions
python -m unittest test_model.py
```

**Test Coverage:**
- Text cleaning functionality
- Label mapping accuracy
- Dataset loading and splitting
- Model prediction pipeline

## 🎯 Usage Example

```python
from Utils import predict_sentiment
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load trained model
model = BertForSequenceClassification.from_pretrained('./models/')
tokenizer = BertTokenizer.from_pretrained('./models/tokenizer/')

# Predict sentiment
review = "This movie was absolutely fantastic! The acting was superb."
sentiment = predict_sentiment(model, tokenizer, device, 128, review)
print(f"Sentiment: {sentiment}")  # Output: Sentiment: Positive
```

## 🚀 Model Deployment

The trained model can be deployed using:

- **Hugging Face Hub**: For easy sharing and inference
- **TorchScript**: For production deployment
- **ONNX**: For cross-platform compatibility
- **Web Interface**: Using Gradio or Streamlit

## 📋 Dependencies

```txt
pandas>=1.5          # Data manipulation
torch>=2.0           # Deep learning framework
transformers>=4.30   # Hugging Face transformers
scikit-learn>=1.3    # ML utilities and metrics
matplotlib>=3.7      # Plotting and visualization
seaborn>=0.12        # Statistical visualization
tqdm>=4.65          # Progress bars
numpy>=1.23         # Numerical computing
mypy>=1.5           # Type checking
flake8>=6.0         # Code style checking
```

## 🔮 Future Enhancements

- **Multi-class Classification**: Extend to neutral/mixed sentiment
- **Model Optimization**: Quantization and pruning for deployment
- **Real-time Inference**: Web API with FastAPI
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Cross-domain Evaluation**: Test on other review types (books, products)
- **Ensemble Methods**: Combine multiple models for better performance

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) creators
- The PyTorch team for the excellent deep learning framework

---

**Built with ❤️ by the Sentiment Analysis Team**
