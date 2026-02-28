import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils import save_model
from utils import setup_logger
import logging

logger = setup_logger()

def clean_data(df):
    """Clean and validate the raw data"""
    try:
        # Clean age column - handle non-numeric values safely
        logger.info("Cleaning age column...")
        df['age'] = df['age'].str.replace("'", "")
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # Fill missing ages with median
        age_median = df['age'].median()
        df['age'] = df['age'].fillna(age_median).astype(int)
        logger.info(f"Median age used for missing values: {age_median}")
        
        # Clean category column
        logger.info("Cleaning category column...")
        df['category'] = df['category'].str.replace("e$_", "").str.replace("*", "")
        
        return df
    except Exception as e:
        logger.error(f"Data cleaning failed: {str(e)}")
        raise

def feature_engineering(df):
    """Create new features using correct column names"""
    try:
        logger.info("Creating same_zip feature...")
        df['same_zip'] = (df['zipcodeOri'] == df['zipMerchant']).astype(int)
        
        logger.info("Creating amount_log feature...")
        df['amount_log'] = np.log1p(df['amount'])
        
        return df
    except KeyError as e:
        logger.error(f"Missing required column: {str(e)}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}")
        raise

def build_preprocessor():
    """Build the sklearn preprocessing pipeline"""
    numeric_features = ['age', 'amount', 'amount_log', 'same_zip']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['gender', 'zipcodeOri', 'zipMerchant', 'category']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

def preprocess_data(input_path, output_path):
    """Main preprocessing function"""
    try:
        logger.info("Loading raw data...")
        df = pd.read_csv(input_path)
        
        # Validate required columns
        required_columns = ['age', 'gender', 'zipcodeOri', 'zipMerchant', 
                          'category', 'amount', 'fraud']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
        logger.info(f"Data shape before preprocessing: {df.shape}")
        logger.info(f"Columns available: {df.columns.tolist()}")
        
        logger.info("Cleaning data...")
        df = clean_data(df)
        
        logger.info("Engineering features...")
        df = feature_engineering(df)
        
        logger.info("Building preprocessing pipeline...")
        preprocessor = build_preprocessor()
        
        logger.info("Transforming data...")
        X = df.drop(['fraud', 'step', 'customer', 'merchant'], axis=1)
        y = df['fraud']
        X_processed = preprocessor.fit_transform(X)
        
        logger.info("Saving processed data...")
        pd.DataFrame(X_processed).to_csv(output_path, index=False)
        save_model(preprocessor, 'models/preprocessing.pkl')
        
        logger.info(f"Preprocessing complete. Final shape: {X_processed.shape}")
        return X_processed, y
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data('data/raw_data.csv', 'data/processed_data.csv')