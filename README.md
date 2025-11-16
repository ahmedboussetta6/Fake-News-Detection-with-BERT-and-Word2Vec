# Fake News Detection with BERT and Word2Vec

A deep learning approach to detect fake news using a hybrid model that combines BERT (Bidirectional Encoder Representations from Transformers) with Word2Vec embeddings for enhanced semantic understanding.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a sophisticated fake news detection system that leverages both BERT's contextual embeddings and Word2Vec's semantic representations. By combining these two powerful NLP techniques, the model achieves exceptional performance in distinguishing between real and fake news articles.

### Key Highlights
- **Hybrid Architecture**: Combines BERT (768-dim) + Word2Vec (300-dim) features
- **High Accuracy**: Achieves ~99.29% accuracy on test set
- **Efficient Training**: 3 epochs with early stopping based on validation loss
- **Balanced Dataset**: Works with both true and fake news samples

## ✨ Features

- **Dual Embedding System**:
  - BERT for contextual word representations
  - Word2Vec (Google News 300) for semantic word embeddings
  
- **Robust Pipeline**:
  - Data preprocessing and tokenization
  - Train/Validation/Test split (70/15/15)
  - Attention masking for variable-length sequences
  
- **Comprehensive Evaluation**:
  - Accuracy and F1-Score metrics
  - Confusion matrix visualization
  - Detailed classification report

## 🏗️ Model Architecture
```
Input Layer
    ↓
[BERT Encoder (768-dim)] + [Word2Vec Features (300-dim)]
    ↓
Concatenation Layer (1068-dim)
    ↓
Dense Layer (512 units) + ReLU + Dropout(0.1)
    ↓
Output Layer (2 classes) + LogSoftmax
```

**Architecture Details**:
- **BERT**: `bert-base-uncased` (frozen pre-trained weights)
- **Word2Vec**: Pre-trained Google News 300-dimensional vectors
- **Optimizer**: AdamW with learning rate 1e-5
- **Loss Function**: Negative Log Likelihood Loss
- **Max Sequence Length**: 15 tokens

## 📊 Dataset

**Source**: [Kaggle - Fake News Detection Dataset](https://www.kaggle.com/code/therealsampat/fake-news-detection/input)

**Structure**:
- `True.csv`: Authentic news articles
- `Fake.csv`: Fake news articles

**Features**:
- `title`: News headline
- `text`: Article content
- `subject`: News category
- `date`: Publication date

**Dataset Split**:
- Training: 70% (31,428 samples)
- Validation: 15% (6,735 samples)
- Test: 15% (6,735 samples)

## 🚀 Installation & Setup

### Prerequisites
This project is designed to run on **Google Colab** with GPU acceleration enabled.

### Steps

1. **Clone the repository**:
```bash
git clone https://github.com/ahmedboussetta6/fake-news-detection-bert-word2vec.git
cd fake-news-detection-bert-word2vec
```

2. **Upload to Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/)
   - Upload `Fake_News_Detection_using_BERT_+_Word2Vec.ipynb`
   - Enable GPU: `Runtime` → `Change runtime type` → `Hardware accelerator: GPU`

3. **Mount Google Drive** (for dataset storage):
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. **Install Dependencies**:
The notebook automatically installs required packages:
```python
!pip install transformers pycaret torch pandas numpy scikit-learn gensim
```

## 💻 Usage

### Running the Complete Pipeline

1. **Load the notebook** in Google Colab

2. **Update dataset paths** in the notebook:
```python
fake_path = 'your_drive_path/Fake.csv'
true_path = 'your_drive_path/True.csv'
```

3. **Download Word2Vec model** (runs once, ~1.6GB):
```python
import gensim.downloader as api
word2vec_model = api.load('word2vec-google-news-300')
word2vec_model.save("word2vec.model")
```

4. **Execute all cells** sequentially to:
   - Load and preprocess data
   - Train the model (3 epochs)
   - Evaluate on test set
   - Generate confusion matrix

### Training Configuration
```python
# Hyperparameters
MAX_LENGTH = 15          # Maximum sequence length
BATCH_SIZE = 32          # Training batch size
LEARNING_RATE = 1e-5     # AdamW learning rate
EPOCHS = 3               # Number of training epochs
DROPOUT = 0.1            # Dropout rate
```

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.29%** |
| **F1-Score** | **99.29%** |
| **Precision (True)** | 99% |
| **Precision (Fake)** | 100% |
| **Recall (True)** | 100% |
| **Recall (Fake)** | 99% |

### Classification Report
```
              precision    recall  f1-score   support

        True       0.99      1.00      0.99      3212
        Fake       1.00      0.99      0.99      3523

    accuracy                           0.99      6735
   macro avg       0.99      0.99      0.99      6735
weighted avg       0.99      0.99      0.99      6735
```

### Confusion Matrix

The model demonstrates excellent performance with minimal misclassifications:
- True Positives: Very high
- False Positives/Negatives: Minimal

*(Visualization available in the notebook)*

### Training Progress

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 0.453         | 0.272          |
| 2     | 0.183         | 0.107          |
| 3     | 0.083         | 0.055          |

## 📁 Project Structure
```
fake-news-detection-bert-word2vec/
│
├── Fake_News_Detection_using_BERT_+_Word2Vec.ipynb    # Main Jupyter notebook with complete pipeline
├── README.md                     # This file
│
└── (Generated files during execution)
    ├── word2vec.model           # Cached Word2Vec model
    ├── best_model_weight.pt     # Saved model weights
    └── bert_tokenizer/          # Saved BERT tokenizer
```

## 🔧 Technical Details

### Key Components

1. **Data Processing**:
   - Tokenization with BERT tokenizer
   - Word2Vec feature extraction
   - Padding to max length (15 tokens)
   - Attention mask generation

2. **Model Features**:
   - Frozen BERT base encoder
   - Word2Vec semantic embeddings
   - Feature concatenation (1068-dim)
   - Two-layer classifier with dropout

3. **Training Strategy**:
   - Best model selection based on validation loss
   - Gradient clipping (max norm = 1.0)
   - Batch-wise progress monitoring

### Hardware Requirements

- **Recommended**: Google Colab with T4 GPU
- **Memory**: ~12GB GPU memory
- **Storage**: ~2GB for models and dataset

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Fake News Detection Dataset](https://www.kaggle.com/code/therealsampat/fake-news-detection/input)
- **BERT Model**: Hugging Face Transformers (`bert-base-uncased`)
- **Word2Vec**: Google News 300-dimensional vectors
- **Framework**: PyTorch
- **Platform**: Google Colab

## 📝 Citation

If you use this project in your research or work, please cite:
```bibtex
@misc{fakenews-bert-word2vec,
  title={Fake News Detection with BERT and Word2Vec},
  author={Ahmed Boussetta},
  year={2025},
  howpublished={\url{https://github.com/ahmedboussetta6/fake-news-detection-bert-word2vec}}
}
```

## 📧 Contact

For questions, suggestions, or collaborations:
- **GitHub**: [@AhmedBoussetta](https://github.com/ahmedboussetta6)
- **Email**: ahmed.boussetta@ensi-uma.tn

## ⭐ Support

If you find this project helpful, please consider giving it a star ⭐ on GitHub!

---

**Note**: This project is for educational and research purposes. The model's predictions should be verified and should not be the sole basis for determining news authenticity.
