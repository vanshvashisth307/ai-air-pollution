import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import logging
from typing import Dict, List, Tuple

# Logging setup
logging.basicConfig(level=logging.INFO)

# Custom CSS for Dark Blue + Sky Blue + White theme
st.markdown("""
    <style>
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
    .main {background-color: #1e3a8a; color: #ffffff; animation: fadeIn 1s;}
    .sidebar .sidebar-content {background-color: #0ea5e9; color: #ffffff; border-radius: 10px;}
    .stButton>button {background-color: #0ea5e9; color: #ffffff; border-radius: 5px; transition: 0.3s;}
    .stButton>button:hover {background-color: #0284c7;}
    .alert-low {color: #10b981; font-weight: bold; animation: fadeIn 0.5s;}
    .alert-moderate {color: #f59e0b; font-weight: bold; animation: fadeIn 0.5s;}
    .alert-high {color: #ef4444; font-weight: bold; animation: fadeIn 0.5s;}
    .tab {border-radius: 10px; padding: 10px; background-color: #ffffff; color: #1e3a8a;}
    h1, h2, h3 {color: #0ea5e9;}
    </style>
    """, unsafe_allow_html=True)

# Hackathon settings
hackathon_end_time = time.time() + 86400  # 24 hours from now; adjust for event

# Backend Functions
def load_data(uploaded_file: st.file_uploader = None) -> pd.DataFrame:
    """
    Load dataset from upload or simulate data.
    
    Args:
        uploaded_file: Uploaded CSV file.
    
    Returns:
        pd.DataFrame: Dataset.
    """
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            logging.info("Dataset loaded from upload.")
            return data
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            st.error("Invalid file. Using simulated data.")
    # Simulate data
    np.random.seed(42)
    data_size = 1000
    data = pd.DataFrame({
        'Temperature': np.random.uniform(-10, 50, data_size),
        'Humidity': np.random.uniform(0, 100, data_size),
        'Wind_Speed': np.random.uniform(0, 50, data_size),
        'Historical_PM25': np.random.uniform(0, 500, data_size),
        'PM25': np.random.uniform(0, 500, data_size)
    })
    logging.info("Simulated dataset created.")
    return data

def train_models(data: pd.DataFrame) -> Tuple[RandomForestRegressor, XGBRegressor, Sequential]:
    """
    Train ML models on the dataset.
    
    Args:
        data: Dataset.
    
    Returns:
        Tuple of trained models.
    """
    X = data[['Temperature', 'Humidity', 'Wind_Speed', 'Historical_PM25']]
    y = data['PM25']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # LSTM
    def prepare_lstm(X, y, time_steps=10):
        X_lstm, y_lstm = [], []
        for i in range(len(X) - time_steps):
            X_lstm.append(X[i:i+time_steps])
            y_lstm.append(y[i+time_steps])
        return np.array(X_lstm), np.array(y_lstm)
    
    X_lstm_train, y_lstm_train = prepare_lstm(X_train.values, y_train.values)
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=0)
    
    logging.info("Models trained successfully.")
    return rf_model, xgb_model, lstm_model

def get_weather_data(city: str, api_key: str) -> Dict[str, float]:
    """
    Fetch weather data from OpenWeatherMap API.
    
    Args:
        city: City name.
        api_key: API key.
    
    Returns:
        Dict of weather data.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'Temperature': data['main']['temp'],
                'Humidity': data['main']['humidity'],
                'Wind_Speed': data['wind']['speed'] * 3.6
            }
        else:
            raise ValueError("API request failed.")
    except Exception as e:
        logging.warning(f"API error: {e}. Using defaults.")
        return {'Temperature': 25, 'Humidity': 60, 'Wind_Speed': 10}

def predict_pm25(model, model_name: str, input_data: pd.DataFrame) -> float:
    """
    Make prediction using selected model.
    
    Args:
        model: Trained model.
        model_name: Name of model.
        input_data: Input features.
    
    Returns:
        Predicted PM2.5.
    """
    if model_name == "LSTM":
        lstm_input = np.array([input_data.values] * 10).reshape(1, 10, 4)
        return model.predict(lstm_input)[0][0]
    return model.predict(input_data)[0]

def calculate_score(predicted: float, actual: float) -> float:
    """
    Calculate hackathon score based on prediction accuracy.
    
    Args:
        predicted: Predicted value.
        actual: Actual value (simulated).
    
    Returns:
        Score (higher is better).
    """
    error = abs(predicted - actual)
    return max(0, 100 - error)  # Simple scoring

# Frontend UI
def main():
    st.markdown("<h1 style='text-align: center; animation: fadeIn 2s;'>🌿 Hackathon: AI Air Pollution Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Predict PM2.5 levels, submit for scoring, and climb the leaderboard! Theme: Dark Blue + Sky Blue + White.")
    
    # Hackathon Timer
    remaining_time = max(0, hackathon_end_time - time.time())
    hours, remainder = divmod(int(remaining_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    st.sidebar.markdown(f"**Hackathon Timer:** {hours:02d}:{minutes:02d}:{seconds:02d}")
    if remaining_time == 0:
        st.sidebar.error("Hackathon ended!")
    
    # Load data and train models (Backend)
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
    data = load_data(uploaded_file)
    rf_model, xgb_model, lstm_model = train_models(data)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔮 Predictions", "📊 Visualizations", "🏆 Leaderboard"])
    
    # Home Tab
    with tab1:
        st.header("Welcome to the Hackathon")
        st.markdown("Upload data, make predictions, and compete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg PM2.5", f"{data['PM25'].mean():.2f}")
        col2.metric("Participants", "Simulated: 50")  # Placeholder
        col3.metric("Top Score", "95.5")
    
    # Predictions Tab
    with tab2:
        st.header("Make & Submit Predictions")
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"])
        use_api = st.checkbox("Use Live Weather")
        city = st.text_input("City", "New York") if use_api else None
        weather = get_weather_data(city, 'YOUR_API_KEY') if use_api else None
        temperature = st.slider("Temperature (°C)", -10, 50, 25 if not weather else weather['Temperature'])
        humidity = st.slider("Humidity (%)", 0, 100, 60 if not weather else weather['Humidity'])
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10 if not weather else weather['Wind_Speed'])
        historical_pm25 = st.slider("Historical PM2.5", 0, 500, 50)
        
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind_Speed': [wind_speed],
            'Historical_PM25': [historical_pm25]
        })
        
        model = {"Random Forest": rf_model, "XGBoost": xgb_model, "LSTM": lstm_model}[model_choice]
        predicted_pm25 = predict_pm25(model, model_choice, input_data)
        
        # Alert
        if predicted_pm25 < 50:
            st.markdown(f"<p class='alert-low'>Predicted: {predicted_pm25:.2f} µg/m³ (Low)</p>", unsafe_allow_html=True)
        elif predicted_pm25 < 150:
            st.markdown(f"<p class='alert-moderate'>Predicted: {predicted_pm25:.2f} µg/m³ (Moderate)</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='alert-high'>Predicted: {predicted_pm25:.2f} µg/m³ (High)</p>", unsafe_allow_html=True)
        
        # Submit for Scoring
        team_name = st.text_input("Team Name")
        if st.button("Submit Prediction"):
            actual_pm25 = np.random.uniform(0, 500)  # Simulate actual
            score = calculate_score(predicted_pm25, actual_pm25)
            if 'leaderboard' not in st.session_state:
                st.session_state.leaderboard = []
            st.session_state.leaderboard.append({'Team': team_name, 'Score': score})
            st.session_state.leaderboard.sort(key=lambda x: x['Score'], reverse=True)
            st.success(f"Submitted! Score: {score:.2f}")
    
    # Visualizations Tab
    with tab3:
        st.header("Visualizations")
        fig = px.scatter(data, x='Temperature', y='PM25', color='Humidity', title="PM2.5 Scatter")
        fig.update_layout(paper_bgcolor='#1e3a8a', plot_bgcolor='#ffffff')
        st.plotly_chart(fig)
    
    # Leaderboard Tab
    with tab4:
        st.header("Leaderboard")
        if 'leaderboard' in st.session_state and st.session_state.leaderboard:
            lb_df = pd.DataFrame(st.session_state.leaderboard)
            st.table(lb_df.head(10))
        else:
            st.info("No submissions yet.")
    
    st.markdown("---")
    st.markdown("Professional code for hackathons. Built with Streamlit.")

if __name__ == "__main__":
    main()