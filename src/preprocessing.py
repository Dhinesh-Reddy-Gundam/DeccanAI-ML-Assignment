import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import fetch_california_housing

def load_data(file_path=None):
    """
    Load the California Housing dataset from scikit-learn or from a CSV file.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, load from scikit-learn.
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    if file_path is None or not os.path.exists(file_path):
        # Load from scikit-learn
        california_housing = fetch_california_housing(as_frame=True)
        df = california_housing.frame
        # Rename columns to match the expected format
        df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
        return df
    else:
        # Load from CSV file
        return pd.read_csv(file_path)

def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        dict: Dictionary containing EDA results
    """
    # Basic information
    info = {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'descriptive_stats': df.describe()
    }
    
    return info

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        pandas.DataFrame: Dataset with handled missing values
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # For numerical columns, fill with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def create_features(df):
    """
    Perform feature engineering on the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        pandas.DataFrame: Dataset with engineered features
    """
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Feature: Rooms per household
    df_new['RoomsPerHousehold'] = df_new['AveRooms'] / df_new['AveOccup']
    
    # Feature: Bedrooms per room
    df_new['BedroomsPerRoom'] = df_new['AveBedrms'] / df_new['AveRooms']
    
    # Feature: Population per household
    df_new['PopulationPerHousehold'] = df_new['Population'] / df_new['AveOccup']
    
    return df_new

def visualize_correlations(df, output_dir='../notebooks'):
    """
    Visualize correlations between features and target variable.
    
    Args:
        df (pandas.DataFrame): Input dataset
        output_dir (str): Directory to save the visualization
        
    Returns:
        None
    """
    # Create correlation matrix
    corr_matrix = df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Create a pairplot for important features
    important_features = ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[important_features])
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()

def preprocess_data(df, target_column='MedHouseVal', test_size=0.2, random_state=42):
    """
    Preprocess the dataset for model training.
    
    Args:
        df (pandas.DataFrame): Input dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create features
    df = create_features(df)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns.tolist()

if __name__ == "__main__":
    # Test the preprocessing functions
    df = load_data('../data/california_housing.csv')
    print("Dataset loaded successfully!")
    
    # Explore the data
    info = explore_data(df)
    print(f"Dataset shape: {info['shape']}")
    print(f"Missing values: {info['missing_values'].sum()}")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Visualize correlations
    visualize_correlations(df)
    print("Correlation visualizations saved to notebooks directory.") 