#!/usr/bin/env python3
"""
Baseline Phishing Detection for AI Inference Endpoints
Author: Krti Tallam
Date: August 14, 2025
Description: Initial Random Forest implementation for phishing detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import time
import joblib
from typing import Dict, List, Tuple
import re
from urllib.parse import urlparse

class AIPhishingDetector:
    """
    Baseline phishing detector specifically designed for AI inference endpoints.
    Based on literature review findings, particularly "Phishing Website Detection 
    using Machine Learning Techniques" (IEEE 2023).
    """
    
    def __init__(self, latency_target_ms: int = 200):
        self.latency_target_ms = latency_target_ms
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,  # Use all cores for speed
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_features(self, request_data: Dict) -> np.ndarray:
        """
        Extract features from API request based on our methodology.
        
        Features include:
        1. URL-based features (from literature)
        2. Payload characteristics
        3. AI-specific patterns
        """
        features = []
        
        # URL-based features
        url = request_data.get('url', '')
        parsed_url = urlparse(url)
        
        # Feature 1: URL length (phishing URLs tend to be longer)
        features.append(len(url))
        
        # Feature 2: Number of dots in URL
        features.append(url.count('.'))
        
        # Feature 3: Contains IP address
        ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        features.append(1 if ip_pattern.search(url) else 0)
        
        # Feature 4: HTTPS usage
        features.append(1 if parsed_url.scheme == 'https' else 0)
        
        # Feature 5: Suspicious keywords in path
        suspicious_keywords = ['admin', 'login', 'verify', 'update', 'security']
        path_lower = parsed_url.path.lower()
        features.append(sum(1 for kw in suspicious_keywords if kw in path_lower))
        
        # Payload features (AI-specific)
        payload = request_data.get('payload', {})
        
        # Feature 6: Payload size (large payloads might be extraction attempts)
        features.append(len(str(payload)))
        
        # Feature 7: Number of nested objects (complexity metric)
        features.append(self._count_nested_depth(payload))
        
        # Feature 8: Contains base64 encoded data
        features.append(1 if self._contains_base64(str(payload)) else 0)
        
        # Feature 9: Request frequency (if available)
        features.append(request_data.get('request_frequency', 0))
        
        # Feature 10: Contains SQL/injection patterns
        injection_patterns = ['select', 'union', 'drop', ';--', 'script>']
        payload_str = str(payload).lower()
        features.append(sum(1 for pattern in injection_patterns if pattern in payload_str))
        
        # AI-specific features
        # Feature 11: Contains model extraction patterns
        extraction_keywords = ['weights', 'parameters', 'architecture', 'layers']
        features.append(sum(1 for kw in extraction_keywords if kw in payload_str))
        
        # Feature 12: Abnormal token length for AI requests
        tokens = request_data.get('tokens', '')
        features.append(len(tokens) if len(tokens) > 100 else 0)
        
        return np.array(features)
    
    def _count_nested_depth(self, obj, depth=0) -> int:
        """Count maximum nesting depth of dictionary/JSON object."""
        if not isinstance(obj, dict):
            return depth
        if not obj:
            return depth
        return max(self._count_nested_depth(v, depth + 1) for v in obj.values())
    
    def _contains_base64(self, text: str) -> bool:
        """Check if text contains base64 encoded data."""
        base64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
        return bool(base64_pattern.search(text))
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train the Random Forest model.
        Returns training metrics.
        """
        start_time = time.time()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        training_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        return {
            'training_time_ms': training_time,
            'training_accuracy': train_accuracy,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
    
    def predict(self, request_data: Dict) -> Tuple[int, float, float]:
        """
        Predict if request is phishing.
        Returns: (prediction, confidence, latency_ms)
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(request_data).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return int(prediction), float(confidence), float(latency_ms)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on test set.
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Measure latency for single prediction
        single_sample = X_test_scaled[0].reshape(1, -1)
        latencies = []
        for _ in range(100):  # Average over 100 predictions
            start = time.time()
            _ = self.model.predict(single_sample)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'latency_target_met': p95_latency < self.latency_target_ms,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_names = saved_data['feature_names']


# Example usage and testing
if __name__ == "__main__":
    print("AI Phishing Detector - Baseline Implementation")
    print("=" * 50)
    
    # Initialize detector
    detector = AIPhishingDetector(latency_target_ms=200)
    
    # Generate synthetic data for demonstration
    # In production, this would use real PhishTank + AI-specific data
    np.random.seed(42)
    n_samples = 1000
    n_features = 12
    
    # Generate synthetic features
    X_synthetic = np.random.randn(n_samples, n_features)
    # Create labels (0: legitimate, 1: phishing)
    y_synthetic = np.random.binomial(1, 0.3, n_samples)  # 30% phishing
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\nTraining Random Forest baseline...")
    train_metrics = detector.train(X_train, y_train)
    print(f"Training completed in {train_metrics['training_time_ms']:.2f}ms")
    print(f"Training accuracy: {train_metrics['training_accuracy']:.3f}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    eval_metrics = detector.evaluate(X_test, y_test)
    print(f"Test accuracy: {eval_metrics['accuracy']:.3f}")
    print(f"F1 score: {eval_metrics['f1_score']:.3f}")
    print(f"Average latency: {eval_metrics['avg_latency_ms']:.2f}ms")
    print(f"95th percentile latency: {eval_metrics['p95_latency_ms']:.2f}ms")
    print(f"Latency target met: {eval_metrics['latency_target_met']}")
    
    # Test single prediction
    print("\nTesting single prediction...")
    test_request = {
        'url': 'https://api.suspicious-ai-service.com/v1/inference',
        'payload': {
            'query': 'extract model weights',
            'nested': {'data': 'test'}
        },
        'tokens': 'a' * 150,
        'request_frequency': 50
    }
    
    prediction, confidence, latency = detector.predict(test_request)
    print(f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Latency: {latency:.2f}ms")
    
    # Save model
    print("\nSaving model...")
    detector.save_model('baseline_model.pkl')
    print("Model saved successfully!")