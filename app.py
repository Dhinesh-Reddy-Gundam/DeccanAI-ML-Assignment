import os
import numpy as np
import pandas as pd
import joblib
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from sklearn.datasets import fetch_california_housing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_fastapi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using a trained XGBoost model",
    version="1.0.0"
)

# Load the model, scaler, and feature names
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
logger.info(MODEL_DIR)

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.joblib'))
    print(f"Model from Joblib is: {model}")
    logger.info(f"Model from Joblib is: {model}")
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    scaler = None
    feature_names = None

# Define Pydantic models for request and response validation
class HousePredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")
    
    @validator('MedInc')
    def validate_income(cls, v):
        if v <= 0:
            raise ValueError("Median income must be positive")
        return v
    
    @validator('AveOccup')
    def validate_occupancy(cls, v):
        if v <= 0:
            raise ValueError("Average occupancy must be positive")
        return v

class PredictionResponse(BaseModel):
    prediction_in_100k: float
    prediction_in_dollars: float
    input_features: Dict[str, float]
    status: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_type: Optional[str] = None
    feature_count: Optional[int] = None

class DatasetInfoResponse(BaseModel):
    name: str
    description: str
    feature_names: List[str]
    target_name: List[str]
    num_samples: int
    num_features: int
    status: str

class ModelInfoResponse(BaseModel):
    model_type: str
    feature_names: List[str]
    feature_count: int
    parameters: Dict[str, Any]
    status: str

@app.get("/", status_code=status.HTTP_200_OK)
def home():
    """
    Home endpoint with general API information.
    """
    return {
        'message': 'House Price Prediction API',
        'status': 'active',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Check API health',
            '/dataset-info': 'GET - Dataset information',
            '/model-info': 'GET - Model information',
            '/docs': 'Interactive API documentation'
        },
        'model': 'XGBoost Regressor (Tuned with GridSearchCV)',
        'dataset': 'California Housing dataset from scikit-learn'
    }

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health():
    """
    Health check endpoint.
    """
    if model is not None and scaler is not None and feature_names is not None:
        status_val = 'healthy'
    else:
        status_val = 'unhealthy'
    
    return {
        'status': status_val,
        'timestamp': datetime.now().isoformat(),
        'model_type': str(type(model).__name__) if model is not None else None,
        'feature_count': len(feature_names) if feature_names is not None else None
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: HousePredictionRequest):
    """
    Prediction endpoint.
    Takes house features as input and returns the predicted house price.
    
    Returns:
        PredictionResponse: Predicted house price in 100k and dollars formats
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None or feature_names is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Convert pydantic model to dict
        data = request.model_dump()
        logger.info(f"Received prediction request: {data}")
        
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])
        
        # Create engineered features
        if 'AveRooms' in input_df.columns and 'AveOccup' in input_df.columns:
            input_df['RoomsPerHousehold'] = input_df['AveRooms'] / input_df['AveOccup']
        
        if 'AveBedrms' in input_df.columns and 'AveRooms' in input_df.columns:
            input_df['BedroomsPerRoom'] = input_df['AveBedrms'] / input_df['AveRooms']
        
        if 'Population' in input_df.columns and 'AveOccup' in input_df.columns:
            input_df['PopulationPerHousehold'] = input_df['Population'] / input_df['AveOccup']
        
        # Ensure the input DataFrame has the same columns as the training data
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features used during training
        input_df = input_df[feature_names]
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Log the prediction
        logger.info(f"Prediction: {prediction}")
        
        # Return the prediction
        return {
            'prediction_in_100k': float(prediction),
            'prediction_in_dollars': float(prediction) * 100000,
            'input_features': data,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/dataset-info", response_model=DatasetInfoResponse, status_code=status.HTTP_200_OK)
def dataset_info():
    """
    Endpoint to provide information about the dataset.
    """
    try:
        # Get dataset information from scikit-learn
        california_housing = fetch_california_housing()
        
        # Fix: Convert feature_names to list if needed
        feature_names_list = california_housing.feature_names
        if hasattr(feature_names_list, 'tolist'):
            feature_names_list = feature_names_list.tolist()
        else:
            feature_names_list = list(feature_names_list)
        
        # Get target_names safely
        if hasattr(california_housing, 'target_names'):
            target_names = california_housing.target_names
            if hasattr(target_names, 'tolist'):
                target_names = target_names.tolist()
            else:
                target_names = list(target_names) if target_names is not None else ["MedHouseVal"]
        else:
            target_names = ["MedHouseVal"]
        
        return {
            'name': 'California Housing Dataset',
            'description': california_housing.DESCR,
            'feature_names': feature_names_list,
            'target_name': target_names,
            'num_samples': california_housing.data.shape[0],
            'num_features': california_housing.data.shape[1],
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dataset information: {str(e)}"
        )

# @app.get("/model-info", response_model=ModelInfoResponse, status_code=status.HTTP_200_OK)
# def model_info():
#     """
#     Endpoint to provide information about the model.
#     """
#     try:
#         if model is None:
#             raise HTTPException(
#                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#                 detail="Model not loaded"
#             )
            
#         # Get model information
#         model_info = {
#             'model_type': str(type(model).__name__),
#             'feature_names': feature_names,
#             'feature_count': len(feature_names) if feature_names is not None else 0,
#             #'parameters': model.get_params(),
#             'status': 'success'
#         }
        
#         return model_info
    
#     except Exception as e:
#         logger.error(f"Error getting model info: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error fetching model information: {str(e)}"
#         )

if __name__ == '__main__':
    # Run the FastAPI app with uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 