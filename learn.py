import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from math import radians, sin, cos, sqrt, atan2

import test

def load_and_preprocess(json_path):
    # Load JSON
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.json_normalize(data)
    
    # Assuming your target variable is 'target'
    df['latitude'] = df['location'].str[0]
    df['longitude'] = df['location'].str[1]
    df.drop(columns='location', inplace=True)
    df['radius'] = df['radius'].str[0]
    # 2024-05-31T17:00:00+00:00
    print(pd.to_datetime((df['effective_from']),format='%Y-%m-%dT%H:%M:%S+00:00'))
    df['start_date'] = pd.to_datetime((df['effective_from']),format='%Y-%m-%dT%H:%M:%S+00:00')
    df['end_date'] = pd.to_datetime((df['effective_to']),format='%Y-%m-%dT%H:%M:%S+00:00')
    
    df.drop(columns=['guid', 'link', 'raw_description', 'publication_date', 'lower_limit', 'upper_limit', 'message', 'effective_from', 'effective_to'], inplace=True)
    X = df.drop(columns='title')
    y = df['title']
    
    print(X)
    
    return X, y

def create_pipeline(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    return pipeline

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def is_location_in_notam(pipeline, notams_data, latitude, longitude):
    for notam in notams_data:
        notam_lat = float(notam['location'][0])
        notam_lon = float(notam['location'][1])
        notam_radius = float(notam['radius'].split(' ')[0]) * 1.852 # Convert nautical miles to km
        
        distance = haversine(latitude, longitude, notam_lat, notam_lon)
        
        if distance <= notam_radius:
            
            input_data = pd.DataFrame([{
                'latitude': latitude,
                'longitude': longitude,
                'radius': notam_radius,
                'start_date': notam['effective_from'],
                'end_date': notam['effective_to']
            }])
            
            input_data['start_date'] = pd.to_datetime((input_data['start_date']),format='%Y-%m-%dT%H:%M:%S+00:00')
            input_data['end_date'] = pd.to_datetime((input_data['end_date']),format='%Y-%m-%dT%H:%M:%S+00:00')
            
            numerical_features = ['latitude', 'longitude', 'radius']
            categorical_features = ['start_date', 'end_date']
            prediction = pipeline.predict(input_data)
            return prediction[0]
    return "No NOTAM found for this location"

def main():
    # Load and preprocess data
    X, y = load_and_preprocess('notams.json')
    
    # Define features
    numerical_features = ['latitude', 'longitude', 'radius']  # Replace with your features
    categorical_features = ['start_date', 'end_date']  # Replace with your features
    
    # Create and train pipeline
    pipeline = create_pipeline(numerical_features, categorical_features)
    pipeline.fit(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #print(X_train)
    print(X_test)
    #print(y_train)
    
    # Evaluate
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # Example usage of the inference function
    notams_data = json.load(open('notams.json', 'r'))
    
    test_latitude = 51.88333333333333
    test_longitude = 0.5666666666666667
    
    #test_latitude = input("Enter Latitude: ")
    #if test_latitude == "":
    #    test_latitude = 51.581
    #test_longitude = input("Enter Longitude: ")
    #if test_longitude == "":
    #    test_longitude = 1.001
    
    result = is_location_in_notam(pipeline, notams_data, float(test_latitude), float(test_longitude))
    print(f"Location ({test_latitude}, {test_longitude}) is in NOTAM: {result}")

if __name__ == "__main__":
    main()
