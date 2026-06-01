import re
import numpy as np
from urllib.parse import urlparse
from typing import Dict, List


class EmailFeatureExtractor:
    """Extract features from emails for phishing detection."""
    
    # Phishing-related keywords
    PHISHING_KEYWORDS = [
        'urgent', 'verify', 'suspend', 'account', 'security', 'update',
        'confirm', 'password', 'login', 'bank', 'paypal', 'immediately',
        'expire', 'restore', 'limited', 'compromised', 'unusual',
        'activity', 'payment', 'invoice', 'claim', 'prize', 'winner',
        'lottery', 'tax', 'irs', 'legal', 'action', 'billing',
        'credit', 'card', 'social', 'security', 'identity', 'theft'
    ]
    
    # Suspicious URL patterns
    SUSPICIOUS_PATTERNS = [
        r'\d+\.\d+\.\d+\.\d+',  # IP address
        r'@[^\s]+',  # @ symbol in URL
        r'[.-]{2,}',  # Multiple dots or dashes
        r'[^a-zA-Z0-9.-]',  # Special characters
    ]
    
    def __init__(self):
        self.phishing_keywords_lower = [kw.lower() for kw in self.PHISHING_KEYWORDS]
    
    def extract_url_features(self, url: str) -> Dict[str, float]:
        """Extract features from a URL."""
        if not url or pd.isna(url):
            return {
                'url_length': 0,
                'has_ip_address': 0,
                'has_at_symbol': 0,
                'has_multiple_dots': 0,
                'has_special_chars': 0,
                'url_entropy': 0,
                'subdomain_count': 0,
                'domain_length': 0
            }
        
        features = {}
        
        # URL length
        features['url_length'] = len(url)
        
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or url
        except:
            domain = url
        
        # Check for IP address
        features['has_ip_address'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # Check for @ symbol
        features['has_at_symbol'] = 1 if '@' in domain else 0
        
        # Check for multiple consecutive dots
        features['has_multiple_dots'] = 1 if '..' in domain else 0
        
        # Check for special characters
        features['has_special_chars'] = 1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0
        
        # Calculate URL entropy (measure of randomness)
        features['url_entropy'] = self._calculate_entropy(url)
        
        # Count subdomains
        subdomains = domain.split('.')
        features['subdomain_count'] = max(0, len(subdomains) - 2)
        
        # Domain length
        features['domain_length'] = len(domain)
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract features from email text."""
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'uppercase_ratio': 0,
                'exclamation_count': 0,
                'phishing_keyword_count': 0,
                'urgent_word_count': 0,
                'money_word_count': 0,
                'has_urgent': 0,
                'has_verify': 0,
                'has_click': 0,
                'has_immediately': 0
            }
        
        features = {}
        text_lower = text.lower()
        words = text_lower.split()
        
        # Text length
        features['text_length'] = len(text)
        
        # Word count
        features['word_count'] = len(words)
        
        # Uppercase ratio
        uppercase_chars = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = uppercase_chars / len(text) if text else 0
        
        # Exclamation mark count
        features['exclamation_count'] = text.count('!')
        
        # Phishing keyword count
        features['phishing_keyword_count'] = sum(1 for kw in self.phishing_keywords_lower if kw in text_lower)
        
        # Urgent words
        urgent_words = ['urgent', 'immediately', 'asap', 'hurry', 'deadline', 'expire']
        features['urgent_word_count'] = sum(1 for word in urgent_words if word in text_lower)
        
        # Money-related words
        money_words = ['money', 'cash', 'payment', 'invoice', 'bank', 'account', 'credit', 'card', 'dollar', '$']
        features['money_word_count'] = sum(1 for word in money_words if word in text_lower)
        
        # Specific keyword flags
        features['has_urgent'] = 1 if 'urgent' in text_lower else 0
        features['has_verify'] = 1 if 'verify' in text_lower else 0
        features['has_click'] = 1 if 'click' in text_lower else 0
        features['has_immediately'] = 1 if 'immediately' in text_lower else 0
        
        return features
    
    def extract_all_features(self, text: str, url: str) -> Dict[str, float]:
        """Extract all features from email text and URL."""
        text_features = self.extract_text_features(text)
        url_features = self.extract_url_features(url)
        
        # Combine features
        all_features = {**text_features, **url_features}
        
        return all_features
    
    def extract_features_batch(self, texts: List[str], urls: List[str]) -> List[Dict[str, float]]:
        """Extract features for a batch of emails."""
        features_list = []
        for text, url in zip(texts, urls):
            features = self.extract_all_features(text, url)
            features_list.append(features)
        return features_list


import pandas as pd
