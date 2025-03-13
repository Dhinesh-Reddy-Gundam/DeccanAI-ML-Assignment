# House Price Prediction Model


## Application link - https://deccanai-ml-assignment-jezzxrzzebah7etuevylps.streamlit

 
This project implements a machine learning model to predict house prices based on various features. It includes data preprocessing, model training, evaluation, and deployment as a REST API with a Streamlit frontend.

## Dataset

The project uses the California Housing dataset from scikit-learn, which is loaded directly using:

```python
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
```

This dataset contains information about houses in California, including features such as median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude. The target variable is the median house value.

## Project Structure

- `models/`: Saved trained models
- `results/`: Contains evaluation metrics and visualizations
- `app_fastapi.py`: FastAPI application for model deployment
- `streamlit_app_simple.py`: Streamlit frontend for interacting with the API (using text input fields)
- `ANALYSIS_REPORT.md`: Detailed analysis of the data and models
- `housing_model_analysis.py`: Script for comprehensive data analysis
- `train_model.py`: Script for training the XGBoost model
- `test_fastapi.py`: Script to test the FastAPI application
- `Dockerfile`: Docker configuration for containerization
- `docker-compose.yml`: Docker Compose configuration for easy deployment

## Features

1. **Comprehensive Data Analysis**:
   - Exploratory data analysis
   - Visualization of distributions and correlations
   - Geographic distribution analysis

2. **Data Preprocessing**:
   - Feature engineering (creating derived features)
   - Outlier handling using winsorization
   - Feature scaling with StandardScaler

3. **Model Training and Evaluation**:
   - Multiple regression models (Linear Regression, Decision Tree, Random Forest, XGBoost)
   - Cross-validation for robust evaluation
   - Evaluation using RMSE, MAE, and R² metrics

4. **Hyperparameter Tuning**:
   - Grid search for XGBoost model
   - Optimized parameters for best performance

5. **Model Deployment**:
   - RESTful API using FastAPI
   - Interactive frontend with Streamlit
   - JSON input/output for easy integration
   - Comprehensive error handling and logging

## Setup and Installation

### Option 1: Local Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model with hyperparameter tuning:
   ```
   python train_model.py
   ```
4. Start the API:
   
   FastAPI (recommended):
   ```
   uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
   ```
   
5. Run the Streamlit frontend:
   ```
   streamlit run streamlit_app_simple.py
   ```

### Option 2: Using Docker

1. Clone this repository
2. Ensure the model is trained and saved in the `models/` directory
3. Build and start the Docker container:
   ```
   docker-compose up -d
   ```

## API Endpoints

The API provides the following endpoints:

1. **/** (Home): General information about the API
2. **/health**: API health check
3. **/predict**: POST endpoint for making predictions
4. **/dataset-info**: Information about the dataset
5. **/model-info**: Information about the trained model

### Making Predictions

Send a POST request to the `/predict` endpoint with the house features in JSON format:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

### Testing the API

Run the test script to verify the API is working correctly:

```bash
python test_fastapi.py
```

## Model Performance

The XGBoost model, after hyperparameter tuning with GridSearchCV, achieves the following performance metrics:

- RMSE: 0.4667
- MAE: 0.3112
- R²: 0.8278

This indicates that the model explains approximately 82.78% of the variance in house prices.

## Feature Importance

The top 5 most important features according to the XGBoost model are:

1. **MedInc** (Median Income): 46.95%
2. **AveOccup** (Average Occupancy): 15.12%
3. **Longitude**: 8.10%
4. **Latitude**: 6.99%
5. **HouseAge**: 5.77%

For a detailed analysis of model performance, feature importance, and methodologies, please refer to the [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) file.

## Frontend UI

The project includes a Streamlit-based user interface for easy interaction with the model:

1. Start the FastAPI server as described above
2. Run the Streamlit app: `streamlit run streamlit_app_simple.py`

