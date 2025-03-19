# Credit Card Fraud Detection

## Overview
This project is a **Credit Card Fraud Detection** system that identifies fraudulent transactions using **unsupervised machine learning techniques**. The system is designed to detect anomalies in transaction patterns using models like **Isolation Forest** and **One-Class SVM**.

## Features
- **Loads and preprocesses** the credit card transaction dataset directly from the web.
- **Trains unsupervised anomaly detection models**:
  - Isolation Forest
  - One-Class SVM
- **Evaluates models** using Accuracy, Precision, Recall, and Confusion Matrix.
- **Visualizes** class distribution and feature correlations.

## Tech Stack
- **Python** (pandas, numpy, scikit-learn, seaborn, matplotlib)
- **Machine Learning Models:** Isolation Forest, One-Class SVM
- **Visualization:** Seaborn, Matplotlib

## Installation
To set up and run the project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
Execute the Python script to detect fraudulent transactions:
```bash
python credit_card_fraud.py
```

## Dataset
The project uses a **Credit Card Transactions Dataset** that contains anonymized transaction details:
- **Amount** (Transaction amount)
- **Time** (Transaction time - removed in preprocessing)
- **V1 to V28** (Anonymized principal components)
- **Class** (0 = Normal, 1 = Fraudulent)

## Project Workflow
1. **Load Data:** Load the dataset from an online source.
2. **Data Preprocessing:**
   - Normalize the transaction amount.
   - Drop unnecessary columns.
   - Split into features (`X`) and labels (`y`).
3. **Train Unsupervised Models:**
   - **Isolation Forest**: Detects anomalies based on transaction distributions.
   - **One-Class SVM**: Classifies normal and fraudulent transactions.
4. **Evaluate Performance:**
   - Compute **Classification Report, Accuracy, Precision, Recall**.
   - Generate **Confusion Matrix**.
5. **Visualization:**
   - Plot class distribution.
   - Display feature correlation heatmap.

## Results
- The **Isolation Forest model** performs well in detecting fraudulent transactions with high precision.
- **Feature importance analysis** reveals key transaction attributes linked to fraud.

## Future Improvements
- Implement **Deep Learning models (Autoencoders)** for enhanced fraud detection.
- Fine-tune model hyperparameters for better precision.
- Deploy the model using **Flask or FastAPI** for real-time fraud detection.

## Author
- Sakshi Kiran Naik
- GitHub: https://github.com/sakshi754/CreditCardFraudDetection/
## License
This project is open-source and available under the MIT License.

