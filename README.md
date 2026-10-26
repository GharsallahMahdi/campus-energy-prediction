# Campus Energy Consumption Prediction

## Project Overview
This project develops an electricity forecasting and analysis system for the STIEI campus in Shanghai.

The STIEI electricity data is simulated. A separate experiment evaluates forecasting models using real historical building electricity data from ASHRAE.

## Main Features
- Interactive campus map and dashboard.
- Detailed monitoring of Computer Building J and Dormitory 9.
- Room-level monitoring for Rooms 9-101 and 9-105.
- 24-hour profiles, 30-day history, and seven-day forecasts.
- Random Forest and LSTM electricity forecasting.
- Building-level and room-level anomaly detection.
- Electricity usage insights: peak hours, load-shifting suggestions, and plain-language summaries.
- Historical Shanghai weather integration using Open-Meteo.
- Independent real-data validation using the ASHRAE dataset.

## AI Insights Module
The current module uses rule-based analysis to identify peak-load periods, suggest energy-saving measures, and explain abnormal electricity consumption.

## Real-Data Experiment
The ASHRAE Great Energy Predictor III experiment evaluates up to 30 educational buildings using Random Forest, LSTM, Gradient Boosting, and baseline models.

The current experiment uses historical consumption lags during testing, so its results are not directly comparable to the recursive seven-day STIEI forecasts. Meter units should be verified before reporting errors as kWh.

## Project Structure
- dashboard/ — interactive campus dashboard.
- src/ — simulation, forecasting, weather, anomaly detection, and insights scripts.
- data/ — generated data, predictions, and evaluation results.
- notebooks/ — experiments.

## Technologies
Python, Pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, HTML, CSS, and JavaScript.

## Run the Dashboard
From the project root, run:

    python3 -m http.server 8000

Then open:

    http://localhost:8000/dashboard/index.html

## Data Sources
- STIEI campus electricity: simulated data.
- Shanghai weather: Open-Meteo historical data, with December 2025 weather used as a proxy for December 2026.
- Real building electricity: ASHRAE Great Energy Predictor III dataset from Kaggle.

The large ASHRAE source datasets are not included in this repository.

