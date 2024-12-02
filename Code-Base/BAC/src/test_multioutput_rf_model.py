import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib
import gc  # Garbage collector for memory optimization

# Suppress warnings to reduce clutter
import warnings
warnings.filterwarnings('ignore')

# Load the preprocessor and model
preprocessor = joblib.load('./models/multioutput_preprocessor.pkl')
multi_output_model = joblib.load('./models/multioutput_rf_model.pkl')

# Load the test dataset
test_data = pd.read_csv('./data/holdout_data_updated.csv')

# Define the feature columns
features = [
    'method', 'status_code', 'response_size', 
    'user_role', 'resource_sensitivity', 'access_type', 
    'is_manipulated', 'is_id_match'
]

# Define the target columns
target = ['bac_vulnerability', 'severity_level', 'priority']

# Split data into features and targets
X_test = test_data[features]
y_test = test_data[target]

# Convert target columns to consistent string types
for col in target:
    y_test[col] = y_test[col].astype(str)

# Preprocess the test data
print("Preprocessing test data...")
X_test_processed = preprocessor.transform(X_test)

# Clear memory
gc.collect()

# Predict on the test data
print("Predicting on test data...")
y_test_pred = multi_output_model.predict(X_test_processed)

# Print evaluation metrics for each target
print("\nTest Data Classification Reports:")
for i, col in enumerate(target):
    print(f"{col} Test Data Classification Report:")
    print(classification_report(y_test[col], y_test_pred[:, i]))
    print(f"{col} Test Data Accuracy Score:", accuracy_score(y_test[col], y_test_pred[:, i]))

# Clear memory after execution
gc.collect()
