import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
    df['start_date'] = pd.to_datetime((df['effective_from']),format='%Y-%m-%dT%H:%M:%S+00:00').to_string()
    df['end_date'] = pd.to_datetime((df['effective_to']),format='%Y-%m-%dT%H:%M:%S+00:00').to_string()
    
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

def main():
    # Load and preprocess data
    X, y = load_and_preprocess('notams.json')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    #print(X_train)
    print(X_test)
    #print(y_train)
    
    # Define features
    numerical_features = ['latitude', 'longitude', 'radius']  # Replace with your features
    categorical_features = ['start_date', 'end_date']  # Replace with your features
    
    # Create and train pipeline
    pipeline = create_pipeline(numerical_features, categorical_features)
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()