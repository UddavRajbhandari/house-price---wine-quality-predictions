import pickle
import pandas as pd
import os

# Define the file paths relative to the script location
model_path = os.path.join(os.path.dirname(__file__), 'q_random_forest_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_q.pkl')

# Load the trained model from the pickle file
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the trained scaler from the pickle file
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

unseen_wine_data = pd.DataFrame({
    'volatile acidity': [0.45],  # Acidity level in the wine
    'residual sugar': [1.345678],  # Sugar left after fermentation
    'chlorides': [0.348912],  # Salt content in the wine
    'sulphates': [0.432876],  # Sulphate level, affects stability
    'alcohol': [1.920345],  # Alcohol content
    'quality': [None],  # Wine quality (used for training, not prediction)
    'fixed_density': [1.702345],  # Wine's density
    'free_total_sulfur': [3.200456],  # Total sulfur dioxide
    'fixed_pH': [2.143678],  # Acidity (pH level)
    'citric_acidity': [1.123456]  # Citric acid content
})

# Convert the dictionary to a pandas DataFrame
unseen_wine_df = pd.DataFrame(unseen_wine_data)

# Ensure the DataFrame has the same columns as the training data
training_columns = ['volatile acidity', 'residual sugar', 'chlorides', 'sulphates', 'alcohol',
                    'fixed_density', 'free_total_sulfur', 'fixed_pH', 'citric_acidity']

unseen_wine_df = unseen_wine_df.reindex(columns=training_columns, fill_value=0)

# Scale the unseen data
unseen_wine_scaled = scaler.transform(unseen_wine_df)

# Predict using the loaded model
predicted_quality = model.predict(unseen_wine_scaled)

# Output the result
print(f"Predicted wine quality: {predicted_quality[0]}")
