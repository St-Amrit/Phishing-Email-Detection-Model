"""
Phishing Email Detection Model - Main Script

This script trains and evaluates a machine learning model for detecting phishing emails.
It uses Scikit-learn's Random Forest classifier with custom feature extraction.
"""

from phishing_detector import PhishingEmailDetector


def main():
    """Main function to train and test the phishing detection model."""
    
    print("=" * 60)
    print("Phishing Email Detection Model")
    print("=" * 60)
    print()
    
    # Initialize detector
    detector = PhishingEmailDetector()
    
    # Train the model
    print("Training the model...")
    print("-" * 60)
    results = detector.train('email_dataset.csv')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Save the model
    detector.save_model('phishing_model.pkl')
    
    # Test with sample emails
    print("\n" + "=" * 60)
    print("Testing with Sample Emails")
    print("=" * 60)
    
    test_emails = [
        {
            'text': 'URGENT: Your account will be suspended. Click here to verify: http://verify-account-now.com',
            'url': 'http://verify-account-now.com',
            'expected': 'Phishing'
        },
        {
            'text': 'Hi team, just a reminder about our meeting tomorrow at 10 AM.',
            'url': '',
            'expected': 'Safe'
        },
        {
            'text': 'Congratulations! You won $1,000,000. Claim now at www.lottery-winner.net',
            'url': 'www.lottery-winner.net',
            'expected': 'Phishing'
        },
        {
            'text': 'Thank you for your recent purchase. Your order has been shipped.',
            'url': '',
            'expected': 'Safe'
        },
        {
            'text': 'Your PayPal account has been limited. Click to restore: http://paypal-secure-login.info',
            'url': 'http://paypal-secure-login.info',
            'expected': 'Phishing'
        }
    ]
    
    print()
    for i, email in enumerate(test_emails, 1):
        result = detector.predict(email['text'], email['url'])
        status = "✓" if result['prediction'] == email['expected'] else "✗"
        
        print(f"Test {i}: {status}")
        print(f"  Text: {email['text'][:60]}...")
        print(f"  URL: {email['url'] if email['url'] else 'None'}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Phishing Probability: {result['phishing_probability']:.2%}")
        print()
    
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Final Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Confusion Matrix:")
    print(results['confusion_matrix'])
    print()
    print("Model saved as 'phishing_model.pkl'")
    print("You can now use the detector.predict() method to classify new emails.")


if __name__ == "__main__":
    main()
