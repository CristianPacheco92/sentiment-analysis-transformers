# Comparative Evaluation of Classical Classifiers and Fine-Tuned DistilBERT for Sentiment Analysis on the IMDb Benchmark

**Coursework COMP1818**  
**Author:** Cristian Pacheco (ID: 001386789)  
**Institution:** School of Computing and Mathematical Sciences, University of Greenwich  

---

## Overview
This repository contains the official implementation and empirical pipeline for document-level binary sentiment classification on the Large Movie Review Dataset (IMDb). The investigation contrasts traditional statistical models (Multinomial Naive Bayes, Linear Support Vector Classification) against a contextualized transformer architecture (DistilBERT fine-tuned on an NVIDIA Tesla T4 GPU).

## Experimental Results (IMDb Test Split - 25,000 samples)

| Model | Accuracy (%) | Macro F1-Score (%) | Training Time |
| :--- | :---: | :---: | :---: |
| Multinomial Naive Bayes | 85.62% | 85.39% | 0.02 s |
| Linear SVM (LinearSVC) | 87.48% | 87.40% | 0.53 s |
| **DistilBERT (Fine-Tuned)** | **91.15%** | **91.22%** | **421.00 s** |

---

## Repository Structure
```text
├── notebooks/
│   └── sentiment_analysis_pipeline.ipynb   # Complete execution pipeline (Colab ready)
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md

---

## How to Run the Code

To ensure an error-free evaluation pipeline as required by the coursework brief:

1. Open the `sentiment_analysis_pipeline.ipynb` file in **Google Colab** (or click the **Open in Colab** badge above).
2. Go to **Runtime** -> **Change runtime type** and select **GPU (T4)**.
3. Run all cells sequentially (**Runtime** -> **Run all**).
4. The notebook will automatically:
   - Install all required dependencies (`transformers`, `datasets`, `accelerate`, `scikit-learn`).
   - Download and preprocess the IMDb dataset (25,000 train / 25,000 test).
   - Train and evaluate the baseline models (TF-IDF + Multinomial Naive Bayes and LinearSVC).
   - Fine-tune and evaluate the DistilBERT model on GPU using FP16 mixed precision.
   - Output the final comparative metrics (Accuracy, Macro F1, training latency) and generate the results figure.
