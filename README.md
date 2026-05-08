# Machine Learning Natural Language Processing Project
Natural Language Processing Project for Machine Learning Academic Course
by Amit Herner & Amit Cohen.
# SMS Spam Classifier: Custom KNN & NLP Pipeline

## Overview
This repository contains a Machine Learning and Natural Language Processing (NLP) project designed to classify SMS messages into **Spam** or **Ham** (legitimate messages). The project features a custom K-Nearest Neighbors (KNN) algorithm implemented entirely from scratch, tailored specifically for text analysis.

## Key Features
* **Custom KNN Algorithm:** Built from scratch using NumPy. Features optimized vectorized operations (avoiding slow `for` loops) and supports both Euclidean and Cosine distances.
* **NLP Feature Engineering:** Text vectorization using `TfidfVectorizer`, including standard English stop-words filtering and N-grams extraction.
* **Handling Imbalanced Data:** Uses `RandomUnderSampler` from the `imblearn` library to balance the dataset and prevent majority-class bias during training.
* **Advanced Pipeline & Tuning:** Integrates TF-IDF and the custom KNN into a scikit-learn `Pipeline`. Optimized using `GridSearchCV` with 5-split `StratifiedKFold` cross-validation.
* **Model Explainability:** Extracts the optimal TF-IDF feature weights to identify and display the top 10 terms that semantically characterize Spam versus Ham messages.

## Results & Evaluation
Our optimized pipeline achieved the following results:
* **Best Parameters:** 3 Neighbors, Cosine Distance, 1000 Max Features, and (1, 2) N-gram range.
* **F1 Macro Score (CV):** 0.926 
* **Precision for Spam:** 0.809 (We heavily prioritized Precision to minimize False Positives and avoid sending legitimate messages to the spam folder).

### Explainability Insights
By analyzing the TF-IDF weights, the model successfully learned the semantic differences between the classes:
* **Top Spam Indicators:** `'www'`, `'ur'`, `'claim'`, `'new'`, `'reply'`, `'free'`
* **Top Ham Indicators:** `'good'`, `'got'`, `'like'`, `'home'`, `'know'`, `'ok'`

## Authors
* **Amit H.**
* **Amit C.**
