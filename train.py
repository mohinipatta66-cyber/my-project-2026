import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # Changed to imblearn's Pipeline
import joblib

def load_data():
    """Load and clean the dataset"""
    try:
        data = pd.read_csv('data/raw_data.csv')
        
        # Clean age column
        data['age'] = (
            data['age']
            .astype(str)
            .str.replace("'", "")
            .apply(lambda x: pd.to_numeric(x, errors='coerce'))
        )
        data['age'] = data['age'].fillna(data['age'].median()).astype(int)
        
        # Clean other columns
        data['category'] = data['category'].str.replace("e$_", "").str.replace("*", "")
        data['fraud'] = data['fraud'].astype(int)
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_features(data):
    """Create essential features"""
    data['amount_log'] = np.log1p(data['amount'])
    data['same_zip'] = (data['zipcodeOri'] == data['zipMerchant']).astype(int)
    return data

def train_and_evaluate():
    """Main training pipeline"""
    # Load and prepare data
    print("Loading data...")
    data = load_data()
    if data is None:
        return
    
    print("Creating features...")
    data = create_features(data)
    
    # Define features
    numeric_features = ['age', 'amount', 'amount_log', 'same_zip']
    categorical_features = ['gender', 'zipcodeOri', 'zipMerchant', 'category']
    
    # Create preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    X = data.drop(['fraud', 'step', 'customer', 'merchant'], axis=1)
    y = data['fraud']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipeline with SMOTE - using imblearn's Pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fraud_model.pkl')
    print("\nModel saved successfully in 'models/' directory")

if __name__ == "__main__":
    print("Fraud Detection System - Training Pipeline")
    print("=========================================")
    train_and_evaluate()