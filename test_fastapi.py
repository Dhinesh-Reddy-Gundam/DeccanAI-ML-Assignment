#!/usr/bin/env python
"""
Script to test the FastAPI House Price Prediction API.
"""

import requests
import json
import sys
import time

def test_api(url="http://localhost:8000"):
    """
    Test the FastAPI House Price Prediction API.
    
    Args:
        url (str): Base URL of the API
    """
    print(f"Testing FastAPI API at {url}...")
    
    # Test home endpoint
    try:
        response = requests.get(f"{url}/")
        print("\n1. Home Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing home endpoint: {str(e)}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{url}/health")
        print("\n2. Health Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing health endpoint: {str(e)}")
    
    # Test dataset-info endpoint
    try:
        response = requests.get(f"{url}/dataset-info")
        print("\n3. Dataset Info Endpoint:")
        print(f"Status Code: {response.status_code}")
        # Print a truncated version of the response
        response_json = response.json()
        if 'description' in response_json:
            response_json['description'] = response_json['description'][:200] + "... (truncated)"
        print(f"Response: {json.dumps(response_json, indent=2)}")
    except Exception as e:
        print(f"Error testing dataset-info endpoint: {str(e)}")
    
    # Test model-info endpoint
    try:
        response = requests.get(f"{url}/model-info")
        print("\n4. Model Info Endpoint:")
        print(f"Status Code: {response.status_code}")
        # Print a truncated version of the response
        response_json = response.json()
        if 'parameters' in response_json:
            # Truncate parameters output to avoid overwhelming display
            param_count = len(response_json['parameters'])
            print(f"Response contains {param_count} model parameters (truncated in output)")
            response_json['parameters'] = "(truncated for display)"
        print(f"Response: {json.dumps(response_json, indent=2)}")
    except Exception as e:
        print(f"Error testing model-info endpoint: {str(e)}")
    
    # Test predict endpoint with valid data
    try:
        data = {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
        
        response = requests.post(
            f"{url}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print("\n5. Predict Endpoint (Valid Data):")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing predict endpoint with valid data: {str(e)}")
    
    # Test predict endpoint with validation errors
    try:
        # Negative income should fail validation
        data = {
            "MedInc": -1.0,  # Negative income - should fail validation
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
        
        response = requests.post(
            f"{url}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print("\n6. Predict Endpoint (Validation Error - Negative Income):")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing predict endpoint with validation error: {str(e)}")
    
    # Test predict with different income levels
    try:
        results = []
        
        income_levels = [2.0, 4.0, 6.0, 8.0, 10.0]
        for income in income_levels:
            data = {
                "MedInc": income,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
            
            response = requests.post(
                f"{url}/predict",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                prediction = response.json().get('prediction_in_100k')
                results.append((income, prediction))
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        print("\n7. Income Level Impact on Predictions:")
        print(f"{'Income Level':<15} {'Predicted House Value':<20}")
        print("-" * 35)
        for income, prediction in results:
            print(f"{income:<15.2f} ${prediction * 100000:<20,.2f}")
    except Exception as e:
        print(f"Error testing income impact: {str(e)}")
    
    print("\nAPI testing completed.")
    
    # Print out the API documentation URLs
    print(f"\nAPI Documentation:")
    print(f"- Swagger UI: {url}/docs")
    print(f"- ReDoc: {url}/redoc")

if __name__ == "__main__":
    # Get the URL from command line arguments if provided
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(url) 