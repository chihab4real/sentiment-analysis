# NLP Project – Sentiment Analysis with BERT on Movie Reviews

## Project Structure

```
├── data/
│   └── imdb_dataset.csv
├── cleaned_splitted_data/
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   └── test_dataset.csv
├── models/
│   ├── sentiment_analysis_model_<timestamp>.pth
│   ├── tokenizer/
│   └── model_config.json
├── Utils.py
├── train_model.ipynb
├── test_model.ipynb
└── README.md
```

## Team Members

* [Chihabeddine Zitouni](https://github.com/chihab4real)
* [Patrick Molina](https://github.com/patrickmolina1/)
* [Małgorzata Gierdewicz](https://github.com/malgier01)



## Dataset

**Source**: [IMDB 50K Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
Balanced dataset with 25,000 positive and 25,000 negative reviews, labeled for binary sentiment classification.


## Setup & Configuration

* Python ≥ 3.8
* PyTorch ≥ 1.10
* HuggingFace Transformers
* Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, tqdm
* Static Analysis: `flake8`, `mypy`

Install requirements:

```bash
pip install -r requirements.txt
```



## Preprocessing & Data Handling

1. **Load & Clean Data**

   * HTML tags and special characters removed
   * Lowercasing and normalization

2. **Label Mapping**

   * `positive` → `1`, `negative` → `0`

3. **Train/Validation/Test Split**

   * 80/10/10 stratified split
   * Stored in `cleaned_splitted_data/`



## Model Architecture

* **Model**: `bert-base-uncased` from HuggingFace
* **Head**: Classification head with 2 output classes
* **Tokenizer**: `BertTokenizer` with `max_length=128`



## Training

* Optimizer: `AdamW`
* Epochs: 3
* Batch Size: 32
* Learning Rate: 2e-5
* Framework: PyTorch

Training and validation loop includes:

* Per-batch loss calculation
* Epoch-wise accuracy tracking
* Real-time progress bars using `tqdm`



## Evaluation

On the test set:

* **Accuracy**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**
* **Loss Curve**
* **Heatmap of Metrics**


## Prediction Example

Custom review sentiment prediction:

```python
"This movie was absolutely fantastic! The acting was superb and the story was captivating."  
→ Sentiment: Positive
```


## Testing

### Unit Tests

Implemented with `unittest` for:

* HTML/text cleaning
* Label mapping functions

### Static Analysis

Run `flake8` and `mypy` to ensure:

* Code style consistency
* Type safety

```bash
flake8 Utils.py train_model.py
mypy Utils.py train_model.py
```

### Model Robustness

The model is further tested for edge cases:

* Evaluates incorrect predictions
* Outputs failing examples for review



## Model Saving & Export

The model and tokenizer are saved to the `models/` directory with timestamped filenames. A `model_config.json` is also exported containing:

* Model file name
* Training parameters
* Final accuracy and F1 score


## How to Run

1. **Train the Model**

   ```bash
   python train_model.py
   ```

2. **Test the Model**

   ```bash
   python test_model.py
   ```

3. **Check Style**

   ```bash
   flake8 Utils.py train_model.py
   ```

4. **Run Type Checks**

   ```bash
   mypy Utils.py train_model.py
   ```


## Future Improvements

* Hyperparameter tuning
* Model quantization for deployment
* Expand dataset with neutral class
* Web interface using Gradio or Streamlit


## License

This project is released for academic use. Please reference the original dataset source when using it in your own work.
