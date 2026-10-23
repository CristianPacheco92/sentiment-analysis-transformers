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
