import streamlit as st
import requests
import json
import pandas as pd


# Configure the page
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint URL - change this if your API is running on a different host/port
API_URL = "http://localhost:8000"

def get_api_status():
    """Check if the API is running and return health status"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def predict_house_price(features):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get('detail', f"Error: Status code {response.status_code}")
            return {"status": "error", "message": error_msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# def get_model_info():
#     """Get model information from API"""
#     try:
#         response = requests.get(f"{API_URL}/model-info")
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return None
#     except Exception:
#         return None

# Helper function to validate numeric input
def is_valid_number(value, min_val=None, max_val=None):
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True
    except ValueError:
        return False

# Main app layout
st.title("🏠 California House Price Predictor")
st.markdown("""
This app predicts house prices in California based on various features.
Enter the details below and click 'Predict' to see the estimated house price.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses a machine learning model to predict house prices in California.
The model is trained on the California Housing dataset.
""")

# Check API status and display
api_status = get_api_status()
if api_status.get("status") == "healthy":
    # st.sidebar.success("✅ API is online and healthy")
    if api_status.get("model_type"):
        st.sidebar.write(f"Model: {api_status.get('model_type')}")
    if api_status.get("feature_count"):
        st.sidebar.write(f"Features: {api_status.get('feature_count')}")
else:
    # st.sidebar.error(f"❌ API is not available: {api_status.get('message', 'Unknown error')}")
    st.error("The prediction service is currently unavailable. Please try again later or contact support.")
    st.stop()

# Display sample values for reference
st.info("""
#### Sample Reference Values:
- Median Income: 8.3 (in tens of thousands)
- House Age: 41 years
- Average Rooms: 7.0
- Average Bedrooms: 1.0
- Population: 322
- Average Occupancy: 2.6
- Latitude: 37.88 (SF: ~37.8, LA: ~34.0)
- Longitude: -122.23 (SF: ~-122.4, LA: ~-118.2)
""")

# Create two columns
col1, col2 = st.columns(2)

# Feature inputs - left column - using text inputs
with col1:
    st.header("Property Features")
    
    med_inc_input = st.text_input(
        "Median Income (tens of thousands $)",
        value="8.3",
        help="Median income of households in the block group"
    )
    med_inc_error = not is_valid_number(med_inc_input, 0.5, 15.0) if med_inc_input else False
    if med_inc_error:
        st.error("Please enter a valid number between 0.5 and 15.0")
    
    house_age_input = st.text_input(
        "House Age (years)",
        value="41",
        help="Median age of houses in the block group"
    )
    house_age_error = not is_valid_number(house_age_input, 1, 60) if house_age_input else False
    if house_age_error:
        st.error("Please enter a valid number between 1 and 60")
    
    ave_rooms_input = st.text_input(
        "Average Rooms",
        value="7.0",
        help="Average number of rooms per household"
    )
    ave_rooms_error = not is_valid_number(ave_rooms_input, 1.0, 12.0) if ave_rooms_input else False
    if ave_rooms_error:
        st.error("Please enter a valid number between 1.0 and 12.0")
    
    ave_bedrms_input = st.text_input(
        "Average Bedrooms",
        value="1.0",
        help="Average number of bedrooms per household"
    )
    ave_bedrms_error = not is_valid_number(ave_bedrms_input, 0.5, 5.0) if ave_bedrms_input else False
    if ave_bedrms_error:
        st.error("Please enter a valid number between 0.5 and 5.0")

# Location inputs - right column - using text inputs
with col2:
    st.header("Location & Population")
    
    population_input = st.text_input(
        "Population",
        value="322",
        help="Population in the block group"
    )
    population_error = not is_valid_number(population_input, 50, 2500) if population_input else False
    if population_error:
        st.error("Please enter a valid number between 50 and 2500")
    
    ave_occup_input = st.text_input(
        "Average Occupancy",
        value="2.6",
        help="Average number of household members"
    )
    ave_occup_error = not is_valid_number(ave_occup_input, 1.0, 6.0) if ave_occup_input else False
    if ave_occup_error:
        st.error("Please enter a valid number between 1.0 and 6.0")
    
    latitude_input = st.text_input(
        "Latitude",
        value="37.88",
        help="Latitude coordinate (higher = more north)"
    )
    latitude_error = not is_valid_number(latitude_input, 32.5, 42.0) if latitude_input else False
    if latitude_error:
        st.error("Please enter a valid number between 32.5 and 42.0")
    
    longitude_input = st.text_input(
        "Longitude",
        value="-122.23",
        help="Longitude coordinate (higher = more east)"
    )
    longitude_error = not is_valid_number(longitude_input, -124.5, -114.0) if longitude_input else False
    if longitude_error:
        st.error("Please enter a valid number between -124.5 and -114.0")
    
    if not latitude_error and not longitude_error:
        try:
            lat_val = float(latitude_input)
            long_val = float(longitude_input)
            st.info(f"Selected location: {lat_val}° N, {long_val}° W")
        except ValueError:
            pass

# Input validation
input_errors = med_inc_error or house_age_error or ave_rooms_error or ave_bedrms_error or population_error or ave_occup_error or latitude_error or longitude_error
has_all_inputs = med_inc_input and house_age_input and ave_rooms_input and ave_bedrms_input and population_input and ave_occup_input and latitude_input and longitude_input

# Prediction section
st.header("Prediction")

# Check for input errors
if not has_all_inputs:
    st.warning("Please fill in all input fields to get a prediction.")
elif input_errors:
    st.error("Please correct the input errors before proceeding.")
else:
    # Convert inputs to float
    med_inc = float(med_inc_input)
    house_age = float(house_age_input)
    ave_rooms = float(ave_rooms_input)
    ave_bedrms = float(ave_bedrms_input)
    population = float(population_input)
    ave_occup = float(ave_occup_input)
    latitude = float(latitude_input)
    longitude = float(longitude_input)
    
    # Create the features dictionary
    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }

    # Display the features data
    st.subheader("Input Features Summary")
    features_df = pd.DataFrame([features])
    st.dataframe(features_df)

    # Predict button
    if st.button("Predict House Price", type="primary"):
        with st.spinner("Predicting..."):
            result = predict_house_price(features)
            
            if result.get("status") == "success":
                # Display prediction
                price_100k = result.get("prediction_in_100k", 0)
                price_dollars = result.get("prediction_in_dollars", 0)
                
                st.success(f"### Predicted House Price: ${price_dollars:,.2f}")
                
                # Show price as a progress bar instead of matplotlib chart
                st.progress(min(price_100k / 8.0, 1.0))  # Scale to max of 800k
                st.caption(f"Price Range: $0 to ${800000:,.2f}")
                
                # Show more details about the prediction
                st.write("#### Property Value Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Price per Room", f"${price_dollars / ave_rooms:,.2f}")
                    st.metric("Price to Income Ratio", f"{price_dollars / (med_inc * 10000):.1f}x")
                
                with col2:
                    st.metric("Price per Bedroom", f"${price_dollars / ave_bedrms:,.2f}")
                    st.metric("Price per Person", f"${price_dollars / ave_occup:,.2f}")
                
            else:
                st.error(f"Prediction failed: {result.get('message', 'Unknown error')}")

# Additional sections - model performance, data insights, etc.
# with st.expander("Model Information"):
#     model_info = get_model_info()
#     if model_info and model_info.get('status') == 'success':
#         st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
#         st.write(f"**Number of Features:** {model_info.get('feature_count', 'Unknown')}")
        
#         # Feature importance as text instead of chart
#         st.write("**Feature Importance:**")
#         st.write("1. **MedInc** (Median Income): 46.95%")
#         st.write("2. **AveOccup** (Average Occupancy): 15.12%")
#         st.write("3. **Longitude**: 8.10%")
#         st.write("4. **Latitude**: 6.99%")
#         st.write("5. **HouseAge**: 5.77%")
#     else:
#         st.write("Model information not available")
    
    # Performance info as text

    # st.write("**Model Performance on Test Data:**")
    # st.write("- RMSE: 0.4667")
    # st.write("- MAE: 0.3112")
    # st.write("- R²: 0.8278")
    
    # st.write("This indicates that the model explains approximately 82.78% of the variance in house prices.")

# Footer
st.markdown("---")
st.markdown("© 2025 House Price Prediction | Created with Streamlit") 