# Tagalog Self-Harm Detection & Safety Routing

An NLP project exploring the detection of **self-harm-related Tagalog prompts** using classical machine learning and transformer-based models, followed by a guarded language-model response layer.

The project compares a **TF-IDF + Linear SVM** baseline with **XLM-RoBERTa**, evaluates performance on an imbalanced dataset, tests robustness against adversarially modified prompts, and explores how the classifier can act as a safety gate before a local LLM.

## Overview

The goal of the project was to build a text-classification pipeline that could distinguish between:

* **0 — Non-self-harm**
* **1 — Self-harm**

Because self-harm prompts made up only a small portion of the original dataset, the project focused not only on accuracy but also on **recall, precision, F1 score, confusion matrices, and ROC-AUC**.

The final workflow also explores how the classifier can sit in front of a generative model, preventing potentially harmful prompts from being passed directly to the LLM.

## Workflow

1. Data loading and class-distribution analysis
2. Random undersampling of the majority class
3. Tagalog and English text preprocessing
4. TF-IDF feature extraction
5. Linear SVM classification
6. XLM-RoBERTa classification
7. Model evaluation and comparison
8. Adversarial robustness testing
9. Safety-routing logic
10. Local LLM integration using OPT-1.3B

## Models

### TF-IDF + Linear SVM

A classical NLP baseline using TF-IDF representations and a linear support vector classifier.

### XLM-RoBERTa

A multilingual transformer model used to better capture contextual information in Tagalog-language prompts.

## Results

| Model       |  Accuracy | Precision |    Recall |        F1 |   AUC-ROC |
| ----------- | --------: | --------: | --------: | --------: | --------: |
| Linear SVM  |     0.772 |     0.101 |     0.946 |     0.182 |     0.856 |
| XLM-RoBERTa | **0.807** | **0.117** | **0.946** | **0.209** | **0.875** |

Both models achieved approximately **94.6% recall** on the self-harm class.

XLM-RoBERTa performed better overall, although precision remained relatively low. This means that while the models detected most positive cases, they also produced a substantial number of false positives.

For a safety-oriented classifier, that trade-off is important: missing a true positive can be costly, but excessive false alarms can also make a system impractical.

## Adversarial Robustness

The project also explored adversarial prompt modification through a pipeline involving translation, text perturbation, and back-translation.

The archived SVM evaluation produced an **attack success rate of 10.81%**, defined as originally correct predictions that became incorrect after adversarial modification.

The original RoBERTa adversarial experiment contained an implementation issue in its comparison setup, so that result is not presented as a valid robustness measurement in the portfolio notebook.

## LLM Safety Routing

A later version of the project integrated a local **OPT-1.3B** language model.

The intended architecture is:

```text
User Prompt
    │
    ▼
Self-Harm Classifier
    │
    ├── Self-Harm Detected ──► Safety Response
    │
    └── Non-Self-Harm ───────► Local LLM
```

This separates **risk detection** from **response generation** rather than relying on the generative model alone to recognize unsafe prompts.

## Tools

`Python` `pandas` `scikit-learn` `TF-IDF` `LinearSVC`
`PyTorch` `Hugging Face Transformers` `XLM-RoBERTa` `OPT-1.3B`
`NLTK` `spaCy`

## Limitations

This project was developed as an academic NLP safety experiment and should not be interpreted as a production-ready self-harm detection system.

Important limitations include:

* a relatively small positive class;
* heavy class imbalance in the original data;
* loss of majority-class information through random undersampling;
* relatively low positive-class precision;
* limited adversarial testing;
* lack of probability calibration and deployment-specific threshold tuning;
* no demographic or linguistic-bias audit; and
* no real-world clinical validation.

A deployed system involving self-harm detection would require substantially broader validation, professional safety review, current crisis-resource information, and careful monitoring.

## Notebook

➡️ [View the complete analysis](tagalog_self_harm_detection_safety_routing.ipynb)

## Data Availability

The original source datasets and fine-tuned XLM-RoBERTa checkpoint are not included in this repository.

The notebook preserves selected outputs from the original execution and is presented as an archived modeling case study rather than a fully reproducible training package.

---

*Originally developed as a group academic project and reorganized for portfolio presentation.*
