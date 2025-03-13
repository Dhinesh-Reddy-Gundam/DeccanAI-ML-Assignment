#!/usr/bin/env python
"""
Quick version of model_tuning.py that runs much faster.
Uses RandomizedSearchCV with fewer iterations and a subset of data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('models', exist_ok=True)

def load_and_preprocess_data_quick():
    """
    Load and preprocess a subset of the California Housing dataset.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("Loading California Housing dataset...")
    
    # Load the dataset
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame
    
    # Rename columns for better readability
    column_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                    'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    df.columns = column_names
    
    # Take a smaller subset for quicker tuning
    df = df.sample(n=3000, random_state=42)
    
    print(f"Using subset of data: {df.shape}")
    
    # Perform feature engineering
    print("\nPerforming feature engineering...")
    df_processed = df.copy()
    
    # Create rooms per household
    df_processed['RoomsPerHousehold'] = df_processed['AveRooms'] / df_processed['AveOccup']
    
    # Create bedrooms per room ratio
    df_processed['BedroomsPerRoom'] = df_processed['AveBedrms'] / df_processed['AveRooms']
    
    # Create population per household
    df_processed['PopulationPerHousehold'] = df_processed['Population'] / df_processed['AveOccup']
    
    # Handle outliers - capping approach
    print("Handling outliers using capping approach...")
    for column in df_processed.columns:
        Q1 = df_processed[column].quantile(0.25)
        Q3 = df_processed[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df_processed[column] = np.where(df_processed[column] < lower_bound, 
                                       lower_bound, df_processed[column])
        df_processed[column] = np.where(df_processed[column] > upper_bound, 
                                       upper_bound, df_processed[column])
    
    # Split features and target
    X = df_processed.drop(columns=['MedHouseVal'])
    y = df_processed['MedHouseVal']
    
    # Train-test split
    print("\nSplitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def tune_xgboost_model_quick(X_train, y_train, X_test, y_test):
    """
    Perform quick hyperparameter tuning for the XGBoost model.
    Uses RandomizedSearchCV with fewer iterations.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
    
    Returns:
        tuple: best_model, best_params, evaluation_metrics
    """
    print("\nPerforming quick hyperparameter tuning for XGBoost...")
    
    # Define a smaller parameter space
    param_dist = {
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'n_estimators': [50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Initial XGBoost model
    xgb = XGBRegressor(random_state=42)
    
    # Randomized search with fewer iterations
    print("Running RandomizedSearchCV with 5 iterations...")
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=5,
        scoring='neg_mean_squared_error',
        cv=3,  # Fewer cross-validation folds
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the random search
    start_time = time.time()
    random_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Get the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    eval_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"\nQuick hyperparameter tuning completed in {tuning_time:.2f} seconds.")
    print(f"Best parameters: {best_params}")
    print(f"Evaluation metrics on test set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return best_model, best_params, eval_metrics

def save_model(model, scaler, feature_names):
    """
    Save the best model, scaler, and feature names.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
    """
    print("\nSaving model, scaler, and feature names...")
    
    # Save model
    model_path = os.path.join('models', 'xgboost_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join('models', 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join('models', 'feature_names.joblib')
    joblib.dump(feature_names, feature_names_path)
    print(f"Feature names saved to {feature_names_path}")

def train_simple_xgboost(X_train, y_train, X_test, y_test):
    """
    Train a simple XGBoost model with default parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
    
    Returns:
        tuple: model, evaluation_metrics
    """
    print("\nTraining a simple XGBoost model (skipping tuning)...")
    
    # Create and train the model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    eval_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"Simple XGBoost training completed in {training_time:.2f} seconds.")
    print(f"Evaluation metrics on test set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return model, eval_metrics

def main():
    """
    Main function to perform quick model training and saving.
    """
    print("Starting quick XGBoost model training and saving...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data_quick()
    
    # Choose one of the following approaches:
    
    # Approach 1: Quick tuning with RandomizedSearchCV
    # best_model, best_params, eval_metrics = tune_xgboost_model_quick(X_train, y_train, X_test, y_test)
    
    # Approach 2: Skip tuning and use a simple model with sensible defaults
    best_model, eval_metrics = train_simple_xgboost(X_train, y_train, X_test, y_test)
    
    # Save the model
    save_model(best_model, scaler, feature_names)
    
    print("\nQuick model training and saving completed successfully!")

if __name__ == "__main__":
    main() 