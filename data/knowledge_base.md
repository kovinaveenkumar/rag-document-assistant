# Machine Learning Concepts Knowledge Base

## Overfitting
Overfitting happens when a model learns the training data too well, including its
noise, and fails to generalize to new data. Signs include high training accuracy
but low validation accuracy. It is reduced with regularization, more data,
cross-validation, dropout, and simpler models.

## Precision and Recall
Precision is the fraction of predicted positives that are actually positive.
Recall is the fraction of actual positives that the model correctly identifies.
In fraud detection, high precision means few false alarms; high recall means few
missed frauds. They trade off, and F1 is their harmonic mean.

## Class Imbalance
Class imbalance occurs when one class vastly outnumbers another, such as fraud
making up under 2% of transactions. Techniques include class weighting,
oversampling (SMOTE), undersampling, and evaluating with PR-AUC instead of
accuracy.

## Retrieval-Augmented Generation (RAG)
RAG combines a retriever and a generator. The retriever finds relevant documents
from a knowledge base using vector similarity over embeddings; the generator (an
LLM) produces an answer grounded in those retrieved documents. RAG reduces
hallucination by anchoring responses in real sources.

## Embeddings
Embeddings are dense vector representations of text that capture semantic meaning.
Similar texts have vectors that are close together (high cosine similarity).
They power semantic search, retrieval, clustering, and recommendation systems.

## Cross-Validation
Cross-validation splits data into k folds, training on k-1 and validating on the
remaining fold, rotating through all folds. It gives a more reliable estimate of
model performance than a single train/test split and helps detect overfitting.

## Gradient Boosting
Gradient boosting builds an ensemble of decision trees sequentially, where each
tree corrects the errors of the previous ones. Popular implementations include
XGBoost, LightGBM, and scikit-learn's HistGradientBoosting. It performs strongly
on tabular data.
