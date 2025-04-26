import pickle
import pandas as pd
import os

from sklearn import metrics
import evaluation

# Define the file paths relative to the script location
model_path = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Load the trained model from the pickle file
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the trained scaler from the pickle file
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


test_df  = pd.read_csv("D:/data science/Challenge accepted/housing_price_prediction/housing-test-set.csv")




# # Define the features for the unseen test data
# unseen_house_data = {
#     'area': [2000],  # Example: 2000 sqft
#     'bedrooms': [3],  # Example: 3 bedrooms
#     'bathrooms': [2],  # Example: 2 bathrooms
#     'stories': [2],  # Example: 2 stories
#     'parking': [1],  # Example: 1 parking space
#     'price': [None],  # 'price' is the target variable; it's unknown for prediction
#     'mainroad_yes': [1],  # Example: House is on the main road
#     'guestroom_yes': [0],  # Example: No guest room
#     'basement_yes': [0],  # Example: No basement
#     'airconditioning_yes': [1],  # Example: Has air conditioning
#     'prefarea_yes': [0],  # Example: Not in a preferred area
#     'furnishingstatus_semi-furnished': [0],  # Example: Not semi-furnished
#     'furnishingstatus_unfurnished': [1],  # Example: Unfurnished
#     'area_bedrooms': [2000 * 3],  # Interaction feature: area * bedrooms
#     'bathrooms_stories': [2 * 2],  # Interaction feature: bathrooms * stories
#     'total_rooms': [3 * 2]  # Total rooms: bedrooms * bathrooms
# }

# # Convert the dictionary to a pandas DataFrame
# unseen_house_df = pd.DataFrame(unseen_house_data)

# # Ensure that the DataFrame has the same columns as the training data
# training_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad_yes', 
#                    'guestroom_yes', 'basement_yes', 'airconditioning_yes', 'prefarea_yes', 
#                    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished', 
#                    'area_bedrooms', 'bathrooms_stories', 'total_rooms']
# unseen_house_df = unseen_house_df.reindex(columns=training_columns, fill_value=0)

# # Scale the unseen data
# unseen_house_scaled = scaler.transform(unseen_house_df)

# # Predict using the loaded model
# predictions = model.predict(unseen_house_scaled)

# # Output the result
# print(f"Predicted price: {predictions[0]}")
