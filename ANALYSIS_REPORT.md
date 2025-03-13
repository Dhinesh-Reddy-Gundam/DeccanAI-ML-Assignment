# California Housing Price Prediction: Model Analysis Report

## 1. Introduction

This report presents the findings from our analysis of the California Housing dataset. We conducted a comprehensive process including data exploration, preprocessing, feature engineering, model training, and evaluation to build a predictive model for housing prices.

## 2. Dataset Overview

The California Housing dataset from scikit-learn contains information about housing in California collected from the 1990 census. It includes 20,640 samples with 8 features:

| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

The target variable is the median house value (MedHouseVal), which is expressed in hundreds of thousands of dollars ($100,000).

## 3. Exploratory Data Analysis

Our exploration revealed several key insights:

1. **Missing Values**: The dataset has no missing values, which simplified our preprocessing.

2. **Feature Correlations**:
   - The most significant correlation with the target variable is median income (0.688).
   - Other important correlations include average rooms (0.152) and house age (0.106).
   - Latitude and Longitude show negative correlations, indicating geographic patterns.

3. **Outliers**:
   - Several features showed outliers, particularly:
     - MedInc: 681 outliers
     - AveRooms: 511 outliers
     - AveBedrms: 1,424 outliers
     - Population: 1,196 outliers
     - AveOccup: 711 outliers
     - MedHouseVal: 1,071 outliers

4. **Geographic Distribution**:
   - Housing prices show clear geographic patterns, with higher values along the coastal areas.
   - The San Francisco Bay Area and parts of Southern California have higher median house values.

## 4. Data Preprocessing

Based on our exploration, we implemented the following preprocessing steps:

1. **Feature Engineering**:
   - Created 'RoomsPerHousehold' = AveRooms / AveOccup
   - Created 'BedroomsPerRoom' = AveBedrms / AveRooms
   - Created 'PopulationPerHousehold' = Population / AveOccup

2. **Handling Outliers**:
   - Used capping approach (winsorization) based on 1.5 × IQR to handle outliers.
   - This preserves the distribution while limiting the impact of extreme values.

3. **Feature Scaling**:
   - Applied StandardScaler to normalize all features to mean=0 and std=1.
   - This ensures all features contribute equally to model performance.

4. **Train-Test Split**:
   - Split data into 80% training and 20% testing sets.
   - Used random_state=42 for reproducibility.

## 5. Model Training and Evaluation

We trained and evaluated four regression models:

1. **Linear Regression**
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**

Each model was evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Training time
- 5-fold cross-validation RMSE

### Model Performance Results

| Model | RMSE | MAE | R² | Training Time (s) |
|-------|------|-----|---|-------------------|
| XGBoost | 0.4667 | 0.3112 | 0.8278 | 0.86 |
| Random Forest | 0.4937 | 0.3261 | 0.8072 | 47.23 |
| Linear Regression | 0.6501 | 0.4762 | 0.6657 | 0.01 |
| Decision Tree | 0.6851 | 0.4469 | 0.6288 | 0.63 |

## 6. Feature Importance Analysis

The top 5 most important features according to the best model (XGBoost) are:

1. **MedInc**: 46.95%
2. **AveOccup**: 15.12%
3. **Longitude**: 8.10%
4. **Latitude**: 6.99%
5. **HouseAge**: 5.77%

This suggests that median income is by far the most important predictor of house prices, followed by household occupancy and geographic location.

## 7. Key Findings and Insights

1. **Model Performance**:
   - XGBoost demonstrated the best overall performance with an R² of 0.8278, meaning it explains approximately 82.78% of the variance in house prices.
   - While Linear Regression is the fastest to train, it performs significantly worse than tree-based models.
   - Random Forest achieves similar performance to XGBoost but takes much longer to train.

2. **Feature Importance**:
   - Median income is the most important predictor across all models, explaining nearly half of the price variation.
   - Geographic location (Latitude and Longitude) plays a significant role, confirming the importance of location in real estate.
   - Our engineered features (RoomsPerHousehold, BedroomsPerRoom) proved to be more valuable than some original features.

3. **Error Analysis**:
   - The best model (XGBoost) achieved an RMSE of 0.4667, meaning predictions are off by about $46,670 on average (since the target is in $100,000s).
   - The MAE of 0.3112 indicates that the typical prediction error is about $31,120.

## 8. Conclusions and Recommendations

Based on our analysis, we recommend:

1. **Model Selection**: 
   - XGBoost should be used for production deployment, as it offers the best balance of accuracy and training time.
   - For scenarios where inference speed is critical, Linear Regression could be considered, though with reduced accuracy.

2. **Feature Engineering**:
   - Focus on income-related features and location data for future model improvements.
   - Consider adding more demographic information and neighborhood characteristics.

3. **Data Collection**:
   - Gather more recent data as the current dataset is from the 1990 census.
   - Include additional features such as proximity to amenities, school quality, and crime rates.

4. **Model Deployment**:
   - The best model should be exposed as a REST API for integration into applications.
   - Implement regular retraining schedules as housing markets change over time.

5. **Future Work**:
   - Explore more complex models such as neural networks for potentially better performance.
   - Implement time-series analysis to capture housing market trends over time.
   - Consider ensemble methods combining multiple models for improved accuracy. 