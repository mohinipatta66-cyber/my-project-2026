import pandas as pd
import numpy as np
import joblib
import os
from typing import Union, Dict, List
from utils import setup_logger

logger = setup_logger()

class FraudDetector:
    def __init__(self, model_dir='models'):
        """Initialize with configurable model directory"""
        self.model_dir = model_dir
        self.models = {
            'preprocessor': None,
            'iso_forest': None,
            'xgb_model': None
        }
        self.threshold = 0.5  # Default decision threshold
        self._load_models()

    def _load_models(self):
        """Load all required models with validation"""
        try:
            logger.info("Loading model artifacts...")
            self.models['preprocessor'] = joblib.load(os.path.join(self.model_dir, 'preprocessing.pkl'))
            self.models['iso_forest'] = joblib.load(os.path.join(self.model_dir, 'iso_forest.pkl'))
            self.models['xgb_model'] = joblib.load(os.path.join(self.model_dir, 'xgb_model.pkl'))
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def set_threshold(self, threshold: float):
        """Update decision threshold (0-1)"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        logger.info(f"Decision threshold updated to {threshold:.2f}")

    def _validate_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Validate input structure and columns"""
        required_cols = {
            'step', 'customer', 'age', 'gender', 
            'zipcodeOri', 'merchant', 'zipMerchant',
            'category', 'amount'
        }

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input must be dict or DataFrame")

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate training data preprocessing"""
        # Clean age (handle both "'3'" and "3" formats)
        df['age'] = df['age'].astype(str).str.replace("'", "").astype(int)
        
        # Clean category (handle both raw and pre-cleaned)
        df['category'] = df['category'].str.replace(r'e\$[_*]*', '', regex=True)
        
        # Feature engineering
        df['same_zip'] = (df['zipcodeOri'] == df['zipMerchant']).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        
        return df

    def predict(self, input_data: Union[Dict, pd.DataFrame], return_features: bool = False):
        """
        Make fraud prediction
        
        Args:
            input_data: Transaction data (dict or DataFrame)
            return_features: Return processed features for debugging
            
        Returns:
            tuple: (prediction, probability) or 
                   (prediction, probability, processed_features)
        """
        try:
            # Validate and clean
            df = self._validate_input(input_data)
            cleaned = self._clean_data(df)
            
            # Prepare features (exclude metadata columns)
            X = cleaned.drop(['step', 'customer', 'merchant'], axis=1)
            X_processed = self.models['preprocessor'].transform(X)
            
            # Get anomaly score
            anomaly_score = self.models['iso_forest'].decision_function(X_processed)
            X_combined = np.column_stack((X_processed, anomaly_score))
            
            # Predict
            proba = self.models['xgb_model'].predict_proba(X_combined)[:, 1][0]
            prediction = int(proba >= self.threshold)
            
            logger.info(f"Prediction: {prediction} (Probability: {proba:.4f}, Threshold: {self.threshold:.2f})")
            
            if return_features:
                return prediction, float(proba), X_combined[0]
            return prediction, float(proba)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

def example_usage():
    """Demo prediction with explanation"""
    detector = FraudDetector()
    
    # Sample transaction matching training format
    transaction = {
        'step': 200,
        'customer': 'C123456789',
        'age': "3'",  # Original format
        'gender': 'M',
        'zipcodeOri': '28007',  # Must match exact column name
        'merchant': 'M987654321',
        'zipMerchant': '28007',
        'category': 'e$_transportation*',  # Raw format
        'amount': 45.67
    }
    
    print("\nSample Transaction:")
    for k, v in transaction.items():
        print(f"{k:>12}: {v}")
    
    # Predict with explanation
    pred, proba, features = detector.predict(transaction, return_features=True)
    
    print("\nPrediction Details:")
    print(f"- Threshold: {detector.threshold:.2f}")
    print(f"- Raw Probability: {proba:.4f}")
    print(f"- Final Prediction: {'FRAUD' if pred else 'LEGITIMATE'}")
    print(f"- Confidence: {'High' if proba > 0.7 or proba < 0.3 else 'Medium'}")

def batch_predict(input_csv: str, output_csv: str, threshold: float = 0.5):
    """Process CSV file with transactions"""
    detector = FraudDetector()
    detector.set_threshold(threshold)
    
    try:
        df = pd.read_csv(input_csv)
        results = []
        
        for _, row in df.iterrows():
            try:
                pred, proba = detector.predict(row.to_dict())
                results.append({
                    'transaction_id': row.get('step', _),
                    'prediction': pred,
                    'probability': proba,
                    'threshold': threshold,
                    'verdict': 'Fraud' if pred else 'Legitimate'
                })
            except Exception as e:
                logger.error(f"Row {_} failed: {str(e)}")
                results.append({
                    'transaction_id': row.get('step', _),
                    'error': str(e)
                })
        
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\nSaved {len(results)} predictions to {output_csv}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Interactive demo
    example_usage()
    
    # Uncomment for batch processing
    # batch_predict('data/new_transactions.csv', 'data/predictions.csv', threshold=0.4)