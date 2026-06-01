# Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as "Phishing" or "Safe" based on textual content and URL features.

## Features

- **Text Analysis**: Extracts features from email text including:
  - Text length and word count
  - Uppercase ratio
  - Exclamation mark count
  - Phishing keyword detection
  - Urgent and money-related word counts

- **URL Analysis**: Extracts features from URLs including:
  - URL length and domain length
  - IP address detection
  - Special character detection
  - Subdomain count
  - URL entropy (randomness measure)

- **Model**: Uses Random Forest classifier for high accuracy
- **Evaluation**: Displays accuracy, confusion matrix, and classification report

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

Run the main script to train and evaluate the model:

```bash
python main.py
```

This will:
- Load the email dataset
- Extract features from emails
- Train a Random Forest classifier
- Display accuracy and confusion matrix
- Save the trained model as `phishing_model.pkl`
- Test with sample emails

### Use the Trained Model

```python
from phishing_detector import PhishingEmailDetector

# Load the trained model
detector = PhishingEmailDetector()
detector.load_model('phishing_model.pkl')

# Predict on a new email
text = "URGENT: Your account will be suspended. Click here to verify: http://verify-account-now.com"
url = "http://verify-account-now.com"

result = detector.predict(text, url)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Phishing Probability: {result['phishing_probability']:.2%}")
```

## Dataset

The `email_dataset.csv` file contains:
- 30 sample emails (15 phishing, 15 safe)
- Columns: `label`, `text`, `urls`
- Labels: `phishing` or `safe`

## Model Performance

The model achieves high accuracy by analyzing:
- Phishing-related keywords (urgent, verify, suspend, account, security, etc.)
- Suspicious URL patterns (IP addresses, special characters, multiple dots)
- Text characteristics (uppercase ratio, exclamation marks, urgency indicators)

## Files

- `main.py` - Main script to train and test the model
- `phishing_detector.py` - ML model class with training and prediction methods
- `feature_extractor.py` - Feature extraction functions for emails and URLs
- `email_dataset.csv` - Sample dataset of phishing and legitimate emails
- `requirements.txt` - Python dependencies
- `phishing_model.pkl` - Saved trained model (generated after running main.py)
