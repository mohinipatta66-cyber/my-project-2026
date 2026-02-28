import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import timedelta
from typing import Union, Dict, List, Tuple
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, 
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.saving import save_detector, load_detector
import shap
import matplotlib.pyplot as plt

class FraudDetectionSystem:
    def __init__(self, model_dir='models'):
        """Initialize the fraud detection system"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {
            'preprocessor': None,
            'anomaly_detector': None,
            'classifier': None,
            'neural_net': None,
            'adversarial_detector': None
        }
        self.threshold = 0.5
        self.explainer = None
        self._load_config()

    def _load_config(self):
        """Load configuration and models"""
        try:
            # Load preprocessing pipeline
            if os.path.exists(os.path.join(self.model_dir, 'preprocessor.pkl')):
                self.models['preprocessor'] = joblib.load(os.path.join(self.model_dir, 'preprocessor.pkl'))
            
            # Load anomaly detector
            if os.path.exists(os.path.join(self.model_dir, 'anomaly_detector')):
                self.models['anomaly_detector'] = load_detector(os.path.join(self.model_dir, 'anomaly_detector'))
            
            # Load classifier
            if os.path.exists(os.path.join(self.model_dir, 'classifier.pkl')):
                self.models['classifier'] = joblib.load(os.path.join(self.model_dir, 'classifier.pkl'))
            
            # Load neural network
            if os.path.exists(os.path.join(self.model_dir, 'neural_net.h5')):
                self.models['neural_net'] = tf.keras.models.load_model(os.path.join(self.model_dir, 'neural_net.h5'))
            
            # Load adversarial detector
            if os.path.exists(os.path.join(self.model_dir, 'adversarial_detector.pkl')):
                self.models['adversarial_detector'] = joblib.load(os.path.join(self.model_dir, 'adversarial_detector.pkl'))
            
        except Exception as e:
            print(f"Error loading models: {e}")

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create innovative features for fraud detection"""
        # Time-based features
        data['hour_of_day'] = data['step'] % 24
        data['is_night'] = ((data['hour_of_day'] >= 22) | (data['hour_of_day'] <= 6)).astype(int)
        
        # Transaction amount features
        data['amount_log'] = np.log1p(data['amount'])
        
        # Location features
        data['same_zip'] = (data['zipcodeOri'] == data['zipMerchant']).astype(int)
        
        # Customer behavior features - CHANGED THIS PART
        data['txn_count_10'] = data.groupby('customer')['step'].transform(
            lambda x: x.rolling(10, min_periods=1).count())
        
        data['avg_amount_ratio'] = data['amount'] / data.groupby('customer')['amount'].transform('mean')
        
        # Merchant risk features
        merchant_fraud_rate = data.groupby('merchant')['fraud'].mean()
        data['merchant_risk'] = data['merchant'].map(merchant_fraud_rate).fillna(0)
        
        return data

    def train_models(self, data_path: str, test_size: float = 0.2):
        """Train the complete fraud detection system"""
        # Load and preprocess data
        data = pd.read_csv(data_path)
        data = self._create_features(data)
        
        # Split data
        X = data.drop(['fraud', 'step', 'customer', 'merchant'], axis=1)
        y = data['fraud']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Preprocessing pipeline
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Train anomaly detector (Isolation Forest + VAE)
        print("Training anomaly detectors...")
        iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
        iso_forest.fit(preprocessor.fit_transform(X_train))
        
        # Train Variational Autoencoder for anomaly detection
        input_dim = preprocessor.transform(X_train[:1]).shape[1]
        vae_detector = OutlierVAE(
            threshold=0.05,  # Adjust based on validation
            encoder_net=Sequential([
                Dense(input_dim, activation='relu'),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(8, activation=None)
            ]),
            decoder_net=Sequential([
                Dense(16, activation='relu'),
                Dense(32, activation='relu'),
                Dense(input_dim, activation=None)
            ]),
            latent_dim=8
        )
        vae_detector.fit(preprocessor.transform(X_train), epochs=50, verbose=0)
        
        # Train ensemble classifier
        print("Training classifier...")
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('oversample', ADASYN(random_state=42)),
            ('classifier', XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0]
        }
        
        gs = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='f1',
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=-1,
            random_state=42
        )
        
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        
        # Train neural network for comparison
        print("Training neural network...")
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(preprocessor.transform(X_train[:1]).shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        nn_model.fit(
            preprocessor.transform(X_train), y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Train adversarial detector
        print("Training adversarial detector...")
        from sklearn.svm import OneClassSVM
        adv_detector = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
        adv_detector.fit(preprocessor.transform(X_train[y_train==0]))
        
        # Evaluate models
        print("\nEvaluating models...")
        self._evaluate_models(best_model, nn_model, preprocessor, X_test, y_test)
        
        # Save models
        self._save_models({
            'preprocessor': preprocessor,
            'anomaly_detector': {'iso_forest': iso_forest, 'vae': vae_detector},
            'classifier': best_model,
            'neural_net': nn_model,
            'adversarial_detector': adv_detector
        })
        
        # Create SHAP explainer
        self._create_explainer(best_model, X_train)
        
    def _evaluate_models(self, model, nn_model, preprocessor, X_test, y_test):
        """Evaluate model performance"""
        # Standard classifier evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print("\n=== Classifier Performance ===")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")
        
        # Neural network evaluation
        nn_prob = nn_model.predict(preprocessor.transform(X_test)).flatten()
        nn_pred = (nn_prob >= 0.5).astype(int)
        
        print("\n=== Neural Network Performance ===")
        print(classification_report(y_test, nn_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, nn_prob):.4f}")
        
        # Find optimal threshold
        self._find_optimal_threshold(y_test, y_prob)
        
    def _find_optimal_threshold(self, y_true, y_prob):
        """Find optimal decision threshold"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"Precision at threshold: {precisions[optimal_idx]:.4f}")
        print(f"Recall at threshold: {recalls[optimal_idx]:.4f}")
        print(f"F1-score at threshold: {f1_scores[optimal_idx]:.4f}")
        
        self.threshold = optimal_threshold
        
    def _save_models(self, models):
        """Save all trained models"""
        joblib.dump(models['preprocessor'], os.path.join(self.model_dir, 'preprocessor.pkl'))
        save_detector(models['anomaly_detector']['vae'], os.path.join(self.model_dir, 'anomaly_detector'))
        joblib.dump(models['anomaly_detector']['iso_forest'], os.path.join(self.model_dir, 'iso_forest.pkl'))
        joblib.dump(models['classifier'], os.path.join(self.model_dir, 'classifier.pkl'))
        models['neural_net'].save(os.path.join(self.model_dir, 'neural_net.h5'))
        joblib.dump(models['adversarial_detector'], os.path.join(self.model_dir, 'adversarial_detector.pkl'))
        
    def _create_explainer(self, model, X_train):
        """Create SHAP explainer for model interpretability"""
        print("\nCreating SHAP explainer...")
        background = X_train.sample(100, random_state=42)
        self.explainer = shap.Explainer(model.named_steps['classifier'], 
                                       model.named_steps['preprocessor'].transform(background))
        
    def predict(self, transaction: Union[Dict, pd.DataFrame], return_details: bool = False):
        """Make fraud prediction with explanation"""
        try:
            # Validate and preprocess input
            if isinstance(transaction, dict):
                df = pd.DataFrame([transaction])
            else:
                df = transaction.copy()
                
            df = self._create_features(df)
            X = df.drop(['step', 'customer', 'merchant'], axis=1, errors='ignore')
            
            # Transform features
            X_processed = self.models['preprocessor'].transform(X)
            
            # Get anomaly scores
            iso_score = self.models['anomaly_detector']['iso_forest'].decision_function(X_processed)
            vae_score = self.models['anomaly_detector']['vae'].score(X_processed)
            
            # Combine features
            X_combined = np.column_stack((X_processed, iso_score, vae_score['data']['instance_score']))
            
            # Predict probability
            proba = self.models['classifier'].predict_proba(X_combined)[:, 1][0]
            prediction = int(proba >= self.threshold)
            
            # Check for adversarial input
            adv_score = self.models['adversarial_detector'].score_samples(X_processed)
            is_adversarial = int(adv_score < -0.5)  # Threshold for adversarial detection
            
            # Generate explanation
            explanation = self._explain_prediction(X) if self.explainer else None
            
            if return_details:
                return {
                    'prediction': prediction,
                    'probability': float(proba),
                    'threshold': self.threshold,
                    'anomaly_scores': {
                        'isolation_forest': float(iso_score[0]),
                        'vae': float(vae_score['data']['instance_score'][0])
                    },
                    'adversarial_flag': is_adversarial,
                    'explanation': explanation
                }
            return prediction, float(proba)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
            
    def _explain_prediction(self, X):
        """Generate SHAP explanation for prediction"""
        shap_values = self.explainer(self.models['preprocessor'].transform(X))
        
        # Visualize explanation
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10)
        plt.tight_layout()
        
        # Return feature importance
        feature_names = self.models['preprocessor'].get_feature_names_out()
        importance = {feature_names[i]: float(shap_values[0].values[i]) 
                     for i in np.argsort(-np.abs(shap_values[0].values))[:10]}
        
        return importance

    def evaluate_adversarial_robustness(self, X_test, y_test, attack_strength=0.1):
        """Evaluate model robustness against adversarial attacks"""
        X_processed = self.models['preprocessor'].transform(X_test)
        
        # Create adversarial examples
        X_adv = X_processed.copy()
        fraud_idx = y_test[y_test == 1].index
        X_adv[fraud_idx] += attack_strength * np.random.randn(len(fraud_idx), X_processed.shape[1])
        
        # Evaluate detection
        adv_scores = self.models['adversarial_detector'].score_samples(X_adv)
        detection_rate = (adv_scores < -0.5).mean()
        
        print(f"\nAdversarial Detection Rate: {detection_rate:.2%}")
        return detection_rate