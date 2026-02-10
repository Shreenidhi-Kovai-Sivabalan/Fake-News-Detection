# Fake News Detection: Neural vs. Traditional Approaches

## Project Overview
This project explores the challenge of automated misinformation detection using Natural Language Processing. I conducted a comparative analysis between traditional machine learning models (Logistic Regression, Naive Bayes, SVM) and deep learning architectures (LSTM, Bidirectional LSTM) to determine the most effective approach for classifying news articles as fake or real.

## Dataset
* **Source:** [Hugging Face: ErfanMoosaviMonazzah/fake-news-detection-dataset-English](https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English)
* **Processing:** The dataset was imported directly via the Hugging Face API and underwent extensive cleaning, including stop-word removal, lemmatization, and n-gram analysis (Bigrams/Trigrams).

## Methodology & Feature Engineering
* **Traditional Models:** Used **TF-IDF Vectorization** to transform text for Logistic Regression, SVM, and Naive Bayes.
* **Deep Learning Models:** Utilized **Word2Vec** and **GloVe** (Global Vectors for Word Representation) embeddings. Applied sequence padding to ensure uniform input length for the neural networks.

## Model Performance
| Model | Feature Extraction | Test Accuracy |
| :--- | :--- | :--- |
| **SVM (Best Baseline)** | TF-IDF | **99.25%** |
| **BiLSTM (Best Neural)** | GloVe (Trainable) | **98.14%** |
| Logistic Regression | TF-IDF | 94.10% |
| Naive Bayes | TF-IDF | 91.50% |

> **Insight:** Interestingly, the SVM model with TF-IDF slightly outperformed the neural approaches in this specific dataset, demonstrating the power of traditional methods on high-quality text features.

## 📂 Repository Structure
* `MainCode.ipynb`: Full pipeline from EDA to Error Analysis.
* `BestTwoModels.ipynb`: Full pipeline for the best two models.
* `Sample_Pred.ipynb`: A lightweight demo notebook to load the pre-trained models and test custom news headlines.
* `models/`: Contains serialized models (`.pkl` and `.h5`) for immediate inference.
