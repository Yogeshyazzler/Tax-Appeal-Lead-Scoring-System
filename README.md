# AI-Powered Tax Appeal Lead Scoring System

## Overview
This project presents a machine learning–driven lead scoring system designed for property tax appeal firms. The system identifies and prioritises property owners most likely to convert into clients by analysing structured property-level data.

Each lead is assigned:
- A calibrated **conversion probability**
- An **expected revenue estimate**
- A **priority segment** for sales action

The solution replaces manual lead qualification with a scalable, data-driven approach.

---

## Key Features

- **Predictive Lead Scoring**  
  Machine learning model estimates the likelihood of client conversion.

- **Lead Segmentation**  
  Leads are categorised into four actionable groups:
  - Call Immediately
  - Nurture
  - Automate
  - Deprioritise

- **Revenue-Based Prioritisation**  
  Expected revenue is calculated to rank leads by business value.

- **Interactive Application**  
  Streamlit interface enables real-time scoring and decision support.

---

## Methodology

### Data Processing
- Input data sourced from Excel files
- Feature engineering includes:
  - Log transformations for skewed variables
  - Frequency and quantile encoding
  - Interaction features

### Model Development
- Primary model: **XGBoost (XGBClassifier)**
- Training approach:
  - Stratified train-validation split
  - Early stopping to prevent overfitting
  - Regularisation tuning

### Imbalance Handling
- **SMOTE** applied when class imbalance exceeds threshold  
- Fallback: `scale_pos_weight` parameter

### Probability Calibration
- **Platt Scaling (CalibratedClassifierCV)** used to ensure predicted probabilities reflect true likelihoods

### Evaluation Metrics
- ROC-AUC
- Precision / Recall
- F1 Score
- Confusion Matrix
- Cross-validation (Stratified K-Fold)

---

## Model Performance

- Strong ROC-AUC on held-out test set  
- Consistent performance across cross-validation folds  
- Well-calibrated probability outputs suitable for business use  

### Key Predictors
- Trestle Score  
- Log-transformed Market Value  
- Feature interaction terms  

---

## Business Impact

- Enables **objective lead prioritisation**
- Improves **sales efficiency and targeting**
- Quantifies opportunity using **expected revenue**
- Reduces reliance on manual or intuition-based processes

---

## Application

The Streamlit application allows users to:
- Input property attributes
- Generate real-time predictions
- View:
  - Conversion probability
  - Expected revenue
  - Lead score
  - Segment classification
  - Recommended action

---

## Tech Stack

| Category            | Tools |
|---------------------|------|
| Language            | Python |
| Data Processing     | Pandas, NumPy |
| Machine Learning    | XGBoost |
| Model Utilities     | Scikit-learn |
| Imbalance Handling  | Imbalanced-learn (SMOTE) |
| Visualisation       | Plotly |
| Web Framework       | Streamlit |

---
