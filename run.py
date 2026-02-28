from fraud_system import FraudDetectionSystem

def main():
    # Initialize the system
    system = FraudDetectionSystem(model_dir='models')
    
    # Train the models (only needed once)
    print("Training models...")
    system.train_models('data/raw_data.csv')
    
    # Example prediction
    sample_transaction = {
        'step': 200,
        'customer': 'C123456789',
        'age': "32",
        'gender': 'M',
        'zipcodeOri': '28007',
        'merchant': 'M987654321',
        'zipMerchant': '28007',
        'category': 'transportation',
        'amount': 150.00
    }
    
    # Make prediction
    result = system.predict(sample_transaction, return_details=True)
    print("\nPrediction Result:")
    for k, v in result.items():
        if k != 'explanation':
            print(f"{k:>20}: {v}")
    
    # Show feature importance
    if result['explanation']:
        print("\nTop Influential Features:")
        for feature, importance in result['explanation'].items():
            print(f"{feature:>30}: {importance:+.4f}")

if __name__ == "__main__":
    main()