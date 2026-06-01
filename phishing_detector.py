import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
from feature_extractor import EmailFeatureExtractor


class PhishingEmailDetector:
    """Machine learning model for phishing email detection."""
    
    def __init__(self):
        self.feature_extractor = EmailFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load email dataset from CSV file."""
        df = pd.read_csv(filepath)
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Extract features and prepare training data."""
        texts = df['text'].tolist()
        urls = df['urls'].tolist()
        
        # Extract features for all emails
        features_list = self.feature_extractor.extract_features_batch(texts, urls)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        # Convert labels to binary (phishing=1, safe=0)
        labels = df['label'].map({'phishing': 1, 'safe': 0}).values
        
        return features_df, labels
    
    def train(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        """Train the phishing detection model."""
        print("Loading data...")
        df = self.load_data(filepath)
        print(f"Dataset loaded with {len(df)} emails")
        print(f"Phishing emails: {sum(df['label'] == 'phishing')}")
        print(f"Safe emails: {sum(df['label'] == 'safe')}")
        
        print("\nExtracting features...")
        features_df, labels = self.prepare_features(df)
        print(f"Extracted {len(self.feature_names)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} emails")
        print(f"Test set: {len(X_test)} emails")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Accuracy percentage: {accuracy * 100:.2f}%")
        
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
    
    def predict(self, text: str, url: str = "") -> dict:
        """Predict if an email is phishing or safe."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(text, url)
        features_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Phishing' if prediction == 1 else 'Safe',
            'confidence': max(probability),
            'phishing_probability': probability[1],
            'safe_probability': probability[0]
        }
        
        return result
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
