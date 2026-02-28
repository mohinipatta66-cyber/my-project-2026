import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_model(model, path):
    import joblib
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

def load_model(path):
    import joblib
    logging.info(f"Loading model from {path}")
    return joblib.load(path)