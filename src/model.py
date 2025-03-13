import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        
    Returns:
        sklearn.linear_model.LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree Regressor model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        
    Returns:
        sklearn.tree.DecisionTreeRegressor: Trained model
    """
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        
    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost Regressor model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        
    Returns:
        xgboost.XGBRegressor: Trained model
    """
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def optimize_random_forest(X_train, y_train, X_test, y_test):
    """
    Optimize a Random Forest Regressor using GridSearchCV.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        
    Returns:
        tuple: (best_model, best_params, evaluation_metrics)
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 15, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create the model
    rf = RandomForestRegressor(random_state=42)
    
    # Create the grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    return best_model, grid_search.best_params_, metrics

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """
    Optimize an XGBoost Regressor using RandomizedSearchCV.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target
        
    Returns:
        tuple: (best_model, best_params, evaluation_metrics)
    """
    # Define the parameter distribution
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # Create the model
    xgb = XGBRegressor(random_state=42)
    
    # Create the randomized search
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    # Fit the randomized search
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Evaluate the best model
    metrics = evaluate_model(best_model, X_test, y_test)
    
    return best_model, random_search.best_params_, metrics

def save_model(model, scaler, feature_names, model_dir=r'../models'):
    """
    Save the trained model and associated objects.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names (list): List of feature names
        model_dir (str): Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save the feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.joblib')
    joblib.dump(feature_names, feature_names_path)
    
    return model_path

def load_model(model_dir=r'../models'):
    """
    Load the trained model and associated objects.
    
    Args:
        model_dir (str): Directory where the model is saved
        
    Returns:
        tuple: (model, scaler, feature_names)
    """
    # Load the model
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    
    # Load the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    # Load the feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.joblib')
    feature_names = joblib.load(feature_names_path)
    
    return model, scaler, feature_names

if __name__ == "__main__":
    # Import preprocessing module
    from preprocessing import load_data, preprocess_data
    
    # Load and preprocess the data
    df = load_data(r'../data/california_housing.csv')
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    print("Training and evaluating models...")
    
    # Train and evaluate Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print(f"Linear Regression - RMSE: {lr_metrics['rmse']:.4f}, MAE: {lr_metrics['mae']:.4f}, R²: {lr_metrics['r2']:.4f}")
    
    # Train and evaluate Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test)
    print(f"Decision Tree - RMSE: {dt_metrics['rmse']:.4f}, MAE: {dt_metrics['mae']:.4f}, R²: {dt_metrics['r2']:.4f}")
    
    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest - RMSE: {rf_metrics['rmse']:.4f}, MAE: {rf_metrics['mae']:.4f}, R²: {rf_metrics['r2']:.4f}")
    
    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost - RMSE: {xgb_metrics['rmse']:.4f}, MAE: {xgb_metrics['mae']:.4f}, R²: {xgb_metrics['r2']:.4f}")
    
    # Optimize Random Forest
    # print("\nOptimizing Random Forest...")
    # best_rf, best_rf_params, best_rf_metrics = optimize_random_forest(X_train, y_train, X_test, y_test)
    # print(f"Best Random Forest - RMSE: {best_rf_metrics['rmse']:.4f}, MAE: {best_rf_metrics['mae']:.4f}, R²: {best_rf_metrics['r2']:.4f}")
    # print(f"Best parameters: {best_rf_params}")
    
    # Optimize XGBoost
    print("\nOptimizing XGBoost...")
    best_xgb, best_xgb_params, best_xgb_metrics = optimize_xgboost(X_train, y_train, X_test, y_test)
    print(f"Best XGBoost - RMSE: {best_xgb_metrics['rmse']:.4f}, MAE: {best_xgb_metrics['mae']:.4f}, R²: {best_xgb_metrics['r2']:.4f}")
    print(f"Best parameters: {best_xgb_params}")
    
    # Save the best model
    # if best_rf_metrics['r2'] > best_xgb_metrics['r2']:
    #     best_model = best_rf
    #     print("\nSaving Random Forest model...")
    # else:
    best_model = best_xgb
    print("\nSaving XGBoost model...")
    
    model_path = save_model(best_model, scaler, feature_names)
    print(f"Model saved to {model_path}") 