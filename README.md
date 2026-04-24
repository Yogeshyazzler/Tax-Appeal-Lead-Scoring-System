📌 Overview

This project delivers an AI-powered lead scoring system for property tax appeal firms, enabling them to identify and prioritise property owners most likely to convert into clients.

By leveraging property-level data such as market value, trestle score, and ownership characteristics, the system predicts conversion probability and assigns a revenue-based priority score to each lead.

🚀 Key Features

✅ Predictive Lead Scoring

Estimates probability of client conversion using machine learning

✅ Actionable Segmentation

Categorises leads into:
🔴 Call Immediately
🟠 Nurture
🔵 Automate
⚪ Deprioritise

✅ Revenue-Based Prioritisation

Expected Revenue =
Conversion Probability × Market Value × Savings Rate × Contingency Fee

✅ Interactive Web App

Real-time scoring via Streamlit interface
Visual insights with dynamic charts
🖼️ System Architecture
🛠️ Tech Stack
Category	Tools Used
Language	Python
Data Processing	Pandas, NumPy
Machine Learning	XGBoost
Model Evaluation	Scikit-learn
Imbalance Handling	SMOTE (Imbalanced-learn)
Visualization	Plotly
Web App	Streamlit
Deployment	Pickle serialization
📊 Model Performance
🎯 Strong ROC-AUC score on held-out test data
🔁 Validated with Stratified K-Fold Cross-Validation
📏 Platt Scaling Calibration ensures reliable probabilities
🔍 Key Predictors
Trestle Score
Log-transformed Market Value
Interaction effects between features
💼 Business Impact

💡 Transforms lead qualification from intuition-based → data-driven
📈 Enables prioritisation based on expected ROI
⚡ Helps sales teams focus on high-value opportunities
📊 Scales across entire prospect databases automatically

Example outputs include:

Conversion Probability
Expected Revenue
Lead Score
Segment Classification

📂 Project Structure
├── data/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgements
Open-source ML ecosystem
Streamlit for rapid UI development
XGBoost for high-performance tabular modelling
