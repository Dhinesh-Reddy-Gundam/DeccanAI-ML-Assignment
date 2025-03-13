#!/usr/bin/env python
"""
Comprehensive script for California Housing price prediction:
1. Data loading and understanding
2. Exploratory data analysis
3. Data preprocessing
4. Model training and evaluation
5. Model comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
import time
import warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('results', exist_ok=True)

def load_and_explore_data():
    """
    Load the California Housing dataset and perform exploratory data analysis.
    
    Returns:
        pandas.DataFrame: The dataset
    """
    print("======== LOADING AND EXPLORING DATA ========")
    print("Loading California Housing dataset...")
    
    # Load the dataset
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame
    
    # Display dataset information
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Rename columns for better readability
    column_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                    'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    df.columns = column_names
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe().round(2))
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:")
    print(missing_values)
    
    # Visualize distributions
    print("\nCreating distribution plots...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/distributions.png')
    plt.close()
    
    # Correlation analysis
    print("\nCreating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('results/correlation_heatmap.png')
    plt.close()
    
    # Display top correlations with target
    target_correlations = correlation_matrix['MedHouseVal'].sort_values(ascending=False)
    print("\nFeature correlations with MedHouseVal (target):")
    print(target_correlations)
    
    # Geographic distribution of housing prices
    print("\nCreating geographic distribution plot...")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                         c=df['MedHouseVal'], cmap='viridis', 
                         s=df['Population']/100, alpha=0.6)
    plt.colorbar(scatter, label='Median House Value')
    plt.title('Geographic Distribution of Housing Prices in California')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('results/geographic_distribution.png')
    plt.close()
    
    # Scatter plots for key features
    print("\nCreating scatter plots for key features...")
    top_features = target_correlations.index[1:4]  # Top 3 features excluding target itself
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, feature in enumerate(top_features):
        sns.regplot(x=feature, y='MedHouseVal', data=df, ax=axes[i])
        axes[i].set_title(f'{feature} vs MedHouseVal')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('MedHouseVal')
    
    plt.tight_layout()
    plt.savefig('results/feature_relationships.png')
    plt.close()
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\n======== PREPROCESSING DATA ========")
    
    # Check for outliers
    print("Checking for outliers...")
    plt.figure(figsize=(12, 8))
    df.boxplot(figsize=(12, 8))
    plt.title('Boxplot of Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/boxplots.png')
    plt.close()
    
    # Identify outliers using IQR
    def identify_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers.shape[0], lower_bound, upper_bound
    
    print("\nOutlier detection:")
    for column in df.columns:
        num_outliers, lower, upper = identify_outliers(df, column)
        if num_outliers > 0:
            print(f"  - {column}: {num_outliers} outliers detected (bounds: {lower:.2f}, {upper:.2f})")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    df_processed = df.copy()
    
    # Create rooms per household
    df_processed['RoomsPerHousehold'] = df_processed['AveRooms'] / df_processed['AveOccup']
    
    # Create bedrooms per room ratio
    df_processed['BedroomsPerRoom'] = df_processed['AveBedrms'] / df_processed['AveRooms']
    
    # Create population per household
    df_processed['PopulationPerHousehold'] = df_processed['Population'] / df_processed['AveOccup']
    
    # Handle outliers - capping approach
    print("\nHandling outliers using capping approach...")
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
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Transform back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate their performance.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
    
    Returns:
        tuple: models, metrics, training_times
    """
    print("\n======== TRAINING AND EVALUATING MODELS ========")
    
    models = {}
    metrics = {}
    training_times = {}
    
    # Define models
    model_constructors = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in model_constructors.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        models[name] = model
        training_times[name] = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Training time: {training_times[name]:.2f} seconds")
        
        # Cross-validation
        print(f"  Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"  Cross-validation RMSE: {cv_rmse:.4f}")
    
    return models, metrics, training_times

def compare_models(metrics, training_times):
    """
    Compare the performance of different models.
    
    Args:
        metrics (dict): Dictionary containing metrics for all models
        training_times (dict): Dictionary containing training times for all models
    """
    print("\n======== MODEL COMPARISON ========")
    
    # Create comparison table
    headers = ["Model", "RMSE", "MAE", "R²", "Training Time (s)"]
    table_data = []
    
    for model_name in metrics:
        table_data.append([
            model_name,
            f"{metrics[model_name]['rmse']:.4f}",
            f"{metrics[model_name]['mae']:.4f}",
            f"{metrics[model_name]['r2']:.4f}",
            f"{training_times[model_name]:.2f}"
        ])
    
    # Sort by R² (descending)
    table_data.sort(key=lambda x: float(x[3]), reverse=True)
    
    print("\nModel Performance Metrics (sorted by R²):")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Find the best model
    best_model = table_data[0][0]
    best_rmse = table_data[0][1]
    best_mae = table_data[0][2]
    best_r2 = table_data[0][3]
    
    print(f"\nBest performing model: {best_model}")
    print(f"  RMSE: {best_rmse}")
    print(f"  MAE: {best_mae}")
    print(f"  R²: {best_r2}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(table_data, columns=headers)
    metrics_df.to_csv('results/model_comparison.csv', index=False)
    print("\nMetrics saved to 'results/model_comparison.csv'")
    
    # Create bar chart for comparison
    plt.figure(figsize=(12, 8))
    
    # Extract model names and metrics
    model_names = [row[0] for row in table_data]
    rmse_values = [float(row[1]) for row in table_data]
    mae_values = [float(row[2]) for row in table_data]
    r2_values = [float(row[3]) for row in table_data]
    
    # Set up bar chart
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bars
    bar1 = ax.bar(x - width, rmse_values, width, label='RMSE', color='cornflowerblue')
    bar2 = ax.bar(x, mae_values, width, label='MAE', color='lightcoral')
    bar3 = ax.bar(x + width, r2_values, width, label='R²', color='mediumseagreen')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Metric Value')
    ax.set_title('Comparison of Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Add values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_chart.png')
    plt.close()
    print("Model comparison chart saved to 'results/model_comparison_chart.png'")

def visualize_feature_importance(models, feature_names):
    """
    Visualize feature importance for tree-based models.
    
    Args:
        models (dict): Dictionary containing trained models
        feature_names (list): List of feature names
    """
    print("\n======== FEATURE IMPORTANCE ANALYSIS ========")
    
    # Filter tree-based models
    tree_models = {name: model for name, model in models.items() 
                   if hasattr(model, 'feature_importances_')}
    
    if not tree_models:
        print("No tree-based models with feature_importances_ attribute found.")
        return
    
    print(f"Analyzing feature importance for {len(tree_models)} tree-based models...")
    
    # Set up the figure
    fig, axes = plt.subplots(len(tree_models), 1, figsize=(12, 5 * len(tree_models)))
    
    # Handle the case when there's only one tree model
    if len(tree_models) == 1:
        axes = [axes]
    
    # Plot for each tree-based model
    for i, (model_name, model) in enumerate(tree_models.items()):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices], ax=axes[i])
        axes[i].set_title(f'{model_name}: Feature Importance')
        axes[i].set_ylabel('Importance')
        axes[i].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # Print feature importance
        print(f"\n{model_name} Feature Importance:")
        for j in indices:
            print(f"  {feature_names[j]}: {importances[j]:.4f}")
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    print("\nFeature importance plot saved to 'results/feature_importance.png'")

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("Starting California Housing Price Prediction Analysis...")
    
    # 1. Load and explore data
    df = load_and_explore_data()
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 3. Train and evaluate models
    models, metrics, training_times = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 4. Compare models
    compare_models(metrics, training_times)
    
    # 5. Analyze feature importance
    visualize_feature_importance(models, feature_names)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 