# EXPOSE: A Comparative Study of Classical and Transformer-Based Approaches for Fake News Detection

## Abstract

The rapid growth of online information platforms has increased the spread of misinformation and fake news. Automated fake news detection has therefore become an important research problem within Natural Language Processing (NLP) and Machine Learning.

This project investigates fake news detection using a progression of increasingly sophisticated approaches, beginning with dataset analysis and classical machine learning methods and culminating in transformer-based language models. Particular emphasis is placed on experimental validity, explainability, and model robustness rather than solely maximizing predictive performance.

The study evaluates Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machines (SVM), and DistilBERT on a large-scale fake news dataset. Additionally, explainability techniques using SHAP and qualitative error analysis are employed to better understand model behavior.

The results demonstrate that both classical and transformer-based methods can achieve extremely high performance on the dataset, with DistilBERT achieving the highest overall accuracy of 99.90%. However, the relatively small performance gap between DistilBERT and Linear SVM highlights the importance of considering computational efficiency alongside predictive performance.

---

# 1. Introduction

The widespread availability of digital media has transformed how information is consumed and shared. While this has increased accessibility to news, it has also enabled misinformation and fabricated content to spread rapidly.

Fake news can influence public opinion, affect political outcomes, and contribute to social instability. Consequently, the development of automated systems capable of distinguishing between legitimate and misleading information has become a significant area of research.

This project aims to investigate the effectiveness of different NLP approaches for fake news detection while addressing several important research questions:

* Can classical machine learning methods effectively detect fake news?
* Do transformer-based language models provide meaningful improvements?
* Which textual patterns influence model decisions?
* What types of articles remain difficult to classify?
* How robust are these systems under imperfect conditions?

---

# 2. Dataset Description

The dataset was obtained through Kaggle and consists of two classes:

* Fake News Articles
* Real News Articles

After combining the source files and assigning labels:

* Label 0 = Fake News
* Label 1 = Real News

The dataset contained approximately 45,000 articles.

Several attributes were available:

* Title
* Article Text
* Subject Category
* Publication Date
* Target Label

Initial exploratory analysis revealed a relatively balanced class distribution, making the dataset suitable for supervised classification.

---

# 3. Dataset Exploration and Leakage Analysis

Before training any models, extensive exploratory data analysis (EDA) was conducted.

The following aspects were investigated:

* Class distribution
* Subject distribution
* Article length statistics
* Duplicate records
* Metadata quality

### Duplicate Analysis

A total of 209 duplicate records were identified and removed to reduce redundancy and prevent biased evaluation.

### Article Length Analysis

Article lengths varied substantially across the dataset, ranging from extremely short reports to long-form news articles.

### Discovery of Data Leakage

One of the most important findings of the project emerged during leakage analysis.

The `subject` feature showed an almost perfect correlation with the target labels:

* Certain subject categories appeared exclusively in fake news articles.
* Other subject categories appeared exclusively in real news articles.

This meant that a model trained on the subject metadata could achieve unrealistically high performance without learning meaningful linguistic patterns.

To preserve experimental validity, metadata fields exhibiting leakage were excluded from all subsequent modeling experiments.

This decision significantly strengthened the credibility of the study.

---

# 4. Classical Machine Learning Baselines

Following the removal of leakage-prone features, textual information was transformed into numerical representations using TF-IDF (Term Frequency–Inverse Document Frequency).

TF-IDF converts text into sparse feature vectors while emphasizing informative words and reducing the influence of extremely common terms.

Three classical machine learning models were evaluated:

### 4.1 Multinomial Naive Bayes

A probabilistic model commonly used in text classification.

Advantages:

* Fast training
* Low memory requirements
* Strong performance on sparse text features

Results:

* Accuracy: 94.51%
* F1 Score: 94.24%

### 4.2 Logistic Regression

A linear classification model frequently used in NLP tasks.

Advantages:

* Interpretable
* Computationally efficient
* Strong baseline performance

Results:

* Accuracy: 98.93%
* F1 Score: 98.87%

### 4.3 Linear Support Vector Machine (SVM)

A maximum-margin classifier particularly effective for high-dimensional sparse data.

Advantages:

* Excellent performance on text classification tasks
* Robust to high-dimensional feature spaces

Results:

* Accuracy: 99.62%
* F1 Score: 99.60%

Among the classical approaches, Linear SVM emerged as the strongest baseline.

---

# 5. Transformer-Based Modeling

To investigate whether contextual language understanding could improve performance further, DistilBERT was evaluated.

### Why DistilBERT?

DistilBERT was selected because:

* It is significantly smaller than BERT.
* It retains most of BERT's predictive performance.
* It requires fewer computational resources.
* It is well-suited for experimentation in Google Colab environments.

### Architecture

DistilBERT is a transformer-based neural network that employs:

* Self-attention mechanisms
* Contextual word embeddings
* Bidirectional language understanding

Unlike TF-IDF methods, transformer models capture semantic relationships between words and account for context during prediction.

### Experimental Setup

* Training Samples: 10,000
* Test Samples: 2,000
* Epochs: 2
* GPU: Tesla T4
* Tokenizer: DistilBERT Tokenizer
* Model: DistilBERT for Sequence Classification

### Results

* Accuracy: 99.90%
* F1 Score: 99.90%

DistilBERT achieved the highest performance of all evaluated models.

---

# 6. Comparative Analysis

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Naive Bayes         | 94.51%   | 94.24%   |
| Logistic Regression | 98.93%   | 98.87%   |
| Linear SVM          | 99.62%   | 99.60%   |
| DistilBERT          | 99.90%   | 99.90%   |

Although DistilBERT achieved the best overall performance, the improvement over Linear SVM was relatively small.

Improvement over Linear SVM:

* Approximately 0.28 percentage points

This finding is important because DistilBERT requires:

* GPU acceleration
* Longer training times
* Higher memory consumption

Consequently, Linear SVM remains a highly attractive solution when computational efficiency is a priority.

---

# 7. Explainability Using SHAP

High performance alone does not guarantee trustworthy predictions.

To improve interpretability, SHAP (SHapley Additive exPlanations) was used to analyze feature importance.

The explainability analysis revealed:

* Words strongly associated with fake news
* Words strongly associated with real news
* Feature contributions to model decisions

This step improved transparency and helped verify that models were learning meaningful textual signals rather than exploiting dataset artifacts.

---

# 8. Error Analysis

The DistilBERT model produced only a small number of errors.

Interestingly:

* All observed errors involved real news articles being classified as fake.
* No fake news articles were incorrectly classified as real.

Inspection of misclassified examples revealed recurring themes:

### Political Content

Several misclassified articles discussed:

* Elections
* Government actions
* Separatist movements
* International conflicts

These topics often contain vocabulary frequently observed in misinformation datasets.

### Human-Interest Stories

Some errors involved short articles with informal or unusual headlines.

These headlines occasionally resembled clickbait-style language commonly associated with fake news.

### Interpretation

The findings suggest that the model relies partially on:

* Stylistic patterns
* Topic-specific vocabulary
* Linguistic framing

While this contributes to strong predictive performance, it can occasionally cause legitimate news articles to be flagged as fake.

---

# 9. Robustness Considerations

Additional experiments explored model robustness through:

### Text Truncation

Articles were shortened to evaluate whether predictions remained stable when contextual information was reduced.

### Noise Injection

Minor spelling perturbations were introduced to simulate noisy user-generated content.

These experiments provided preliminary evidence that model predictions remained relatively stable under small perturbations.

However, more extensive robustness testing would be required before deployment in real-world environments.

---

# 10. Why This Study Was Successful

Several factors contributed to the success of this project:

1. Careful identification and removal of data leakage.
2. Strong baseline comparisons using classical machine learning methods.
3. Evaluation of modern transformer architectures.
4. Inclusion of explainability techniques.
5. Investigation of model errors rather than reporting only accuracy.
6. Consideration of robustness and reliability.

As a result, the project goes beyond simple model training and provides a more complete scientific investigation of fake news detection.

---

# 11. Limitations

Despite strong results, several limitations remain.

### Dataset Bias

The dataset originates from specific sources and time periods.

Models may therefore learn source-specific writing styles rather than universal misinformation characteristics.

### Binary Classification

The task assumes only two categories:

* Fake
* Real

In reality, misinformation exists on a spectrum.

### Limited Generalization Testing

The models were evaluated on data drawn from the same distribution as the training set.

Cross-domain evaluation would provide stronger evidence of real-world effectiveness.

---

# 12. Future Work

Several directions could extend this research:

### Advanced Transformer Models

* RoBERTa
* DeBERTa
* ELECTRA

### Fact-Checking Integration

Combine textual classification with external knowledge sources and fact verification systems.

### Adversarial Robustness

Evaluate performance under deliberate manipulation and adversarial attacks.

### Cross-Domain Evaluation

Assess generalization across:

* Politics
* Health misinformation
* Financial misinformation
* Social media content

### Explainable AI

Investigate advanced explainability frameworks for transformer architectures.

---

# 13. Conclusion

This study explored fake news detection through a comprehensive sequence of experiments involving exploratory analysis, classical machine learning methods, transformer-based architectures, explainability techniques, and robustness evaluation.

The strongest performance was achieved by DistilBERT, reaching 99.90% accuracy and 99.90% F1 score. However, Linear SVM remained highly competitive while requiring substantially fewer computational resources.

The project demonstrates that effective fake news detection depends not only on predictive performance but also on careful dataset analysis, experimental validity, interpretability, and robustness evaluation.

Overall, the findings highlight both the potential and the limitations of modern NLP techniques in addressing misinformation detection challenges.
