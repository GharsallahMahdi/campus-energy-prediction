# Campus Energy Consumption Prediction

## Project Overview

This project develops an electricity consumption forecasting, anomaly detection, and energy analysis system for the **STIEI campus in Shanghai, China**.

The goal is to help campus administrators understand electricity usage patterns, predict future demand, detect unusual consumption, and identify opportunities to reduce energy waste.

**Important:** STIEI campus electricity data and anomaly scenarios are simulated for demonstration purposes. A separate experiment evaluates forecasting models on the real-world ASHRAE Great Energy Predictor III dataset.

## Main Features

- Interactive campus map and electricity monitoring dashboard.
- Detailed monitoring of **Computer Building J** and **Dormitory 9**.
- Room-level monitoring of four selected rooms in Dormitory 9.
- Typical daily electricity profiles by season and weekday/weekend.
- Last 30 days of electricity consumption history.
- Separate 24-hour and 7-day electricity forecasts.
- Random Forest and LSTM forecasting models.
- Forecast evaluation using MAE, RMSE, MAPE, and R².
- Building-level and room-level anomaly detection.
- Estimated extra electricity costs and building monthly bill projections.
- Rule-based energy insights, peak-hour identification, and load-shifting recommendations.
- Historical Shanghai weather integration using Open-Meteo.
- Independent real-data forecasting experiment using the ASHRAE dataset.

## Four-Room Monitoring — Dormitory 9

The dashboard provides a detailed view of four selected rooms representing a dormitory floor.

| Room | Scenario | Description |
|---|---|---|
| 9-101 | Normal | Reference room without injected anomalies |
| 9-105 | AC left on | Simulated abnormal consumption from December 27–29, 2026 |
| 9-110 | Normal | Second reference room without injected anomalies |
| 9-115 | Equipment left running | Simulated overnight abnormal consumption from December 20–24, 2026 |

The anomaly scenarios are deliberately injected into the simulated dataset to test the detection system.

The dashboard allows users to switch between the whole dormitory building and individual rooms.

## Electricity Forecasting

Two machine learning models are implemented:

- **Random Forest**
- **Long Short-Term Memory (LSTM)** neural network

The system produces forecasts for:

1. **Next 24 hours**
2. **Next 7 days (168 hours)**

Forecasting uses historical electricity consumption, calendar information, and weather features, including temperature and humidity.

The forecasts are generated recursively without using future electricity consumption as model input.

### Forecast Evaluation

Models are evaluated separately for each forecast horizon using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **R²:** Coefficient of Determination

### Latest Room Forecast Results

Random Forest MAPE on the simulated STIEI room dataset:

| Room | Next 24 Hours | Next 7 Days |
|---|---:|---:|
| 9-101 | 8.74% | 12.87% |
| 9-105 | 9.84% | 29.19% |
| 9-110 | 11.40% | 12.57% |
| 9-115 | 15.88% | 24.22% |

In the latest four-room experiment, Random Forest achieved lower forecasting errors than LSTM.

Forecasting unexpected anomaly events remains challenging, particularly over longer horizons.

## Typical Electricity Consumption Profiles

The dashboard displays typical 24-hour electricity consumption profiles for:

- Winter, spring, summer, and fall.
- Weekdays and weekends.
- Buildings and selected rooms.

Academic-break periods and deliberately injected anomalies are excluded when calculating typical profiles.

These profiles help identify differences between teaching buildings and dormitories and show how electricity demand changes across seasons.

## Anomaly Detection and Electricity Costs

The system compares observed electricity consumption against typical consumption patterns to identify unusual usage.

### Room-Level Alerts

The latest simulated experiment produced:

| Room | Detected Alerts | Estimated Extra Cost |
|---|---:|---:|
| 9-101 | 0 | ¥0.00 |
| 9-105 | 66 | ¥32.07 |
| 9-110 | 3 | ¥0.44 |
| 9-115 | 25 | ¥7.91 |

Room 9-110 has no injected anomaly but produced three alerts, illustrating the possibility of false positives.

### Cost Estimation

- Estimated extra consumption is calculated relative to a typical baseline.
- Estimated additional costs use an **assumed electricity tariff of ¥0.53/kWh**.
- Building-level monitoring also includes monthly electricity bill projections.
- Cost estimates are illustrative and do not represent verified STIEI electricity bills or tariffs.

## AI Insights Module

The dashboard includes a **rule-based energy insights module** that:

- Identifies typical peak-consumption hours.
- Highlights peak periods in a visual chart.
- Provides practical load-shifting recommendations.
- Summarizes detected anomalies.
- Displays estimated additional electricity costs.

For example, dormitory recommendations may suggest moving flexible electricity use away from the evening peak.

The current module uses programmed analysis rules rather than a generative AI model.

## Weather Data Integration

The project integrates historical hourly Shanghai weather data from **Open-Meteo**.

Weather features include:

- Air temperature
- Relative humidity

Because actual weather observations for the simulated forecast period are unavailable, historical weather from **December 25–31, 2025** is aligned with the simulated forecast period of **December 25–31, 2026** as a proxy.

This is a demonstration approach, not a validated operational weather forecast for December 2026.

## Real-Data Experiment — ASHRAE

A separate experiment uses the **ASHRAE Great Energy Predictor III** dataset to evaluate forecasting models on real educational-building electricity data.

The experiment includes up to 30 selected educational buildings and compares:

- Mean baseline
- Hourly-profile baseline
- Last-week baseline
- Random Forest
- Random Forest with consumption lags
- Gradient Boosting with consumption lags
- LSTM

### Experimental Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 28.865 | 45.184 | 0.980 |
| Hourly-profile baseline | 19.910 | 33.564 | 0.989 |
| Last-week baseline | 13.089 | 27.414 | 0.993 |
| Random Forest (time features) | 16.047 | 28.712 | 0.992 |
| Random Forest + lags | 10.129 | 20.690 | 0.996 |
| Gradient Boosting + lags | 9.949 | 20.441 | 0.996 |
| LSTM | 7.252 | 15.954 | 0.997 |

**Important methodological distinction:** The ASHRAE experiment uses observed historical consumption lags during the test period, so these results are not directly comparable with the recursive seven-day STIEI forecasts.

The large original ASHRAE dataset files are not included in this repository.

## Interactive Dashboard

The dashboard contains:

- A campus map with two interactive building markers.
- A room selector for Dormitory 9.
- Seasonal 24-hour consumption profiles.
- A 30-day consumption history.
- Separate 24-hour and 7-day forecast charts.
- Actual versus predicted electricity consumption.
- A weather temperature overlay on forecast charts.
- Anomaly notifications and estimated extra costs.
- Energy insights and recommendations.

## Project Structure

```text
campus-energy-project/
├── dashboard/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── campus_map.jpg
│   └── data.json
├── src/
│   ├── simulate_data.py
│   ├── simulate_rooms.py
│   ├── forecast_short_term.py
│   ├── forecast_buildings_j9.py
│   ├── forecast_rooms.py
│   ├── anomaly_detection.py
│   ├── detect_room_anomalies.py
│   ├── generate_insights.py
│   ├── fetch_real_weather.py
│   ├── generate_weather.py
│   ├── weather_source.py
│   ├── typical_profiles.py
│   ├── build_map2_data.py
│   └── ashrae_experiment.py
├── data/
├── notebooks/
└── README.md
```

## Technologies

- **Python:** Pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib
- **Frontend:** HTML, CSS, JavaScript, SVG
- **Weather:** Open-Meteo historical weather API
- **Version Control:** Git and GitHub

## Run the Dashboard

From the project root, start a local web server:

```bash
python3 -m http.server 8000
```

Then open the following address in your browser:

http://localhost:8000/dashboard/index.html

The dashboard reads the generated `dashboard/data.json` file.

## Data Sources

- **STIEI campus electricity:** Simulated building-level and room-level data.
- **Shanghai weather:** Historical Open-Meteo data, using December 2025 as a proxy for the simulated December 2026 forecast period.
- **ASHRAE electricity:** Real-world Great Energy Predictor III dataset available from Kaggle.

ASHRAE dataset: https://www.kaggle.com/competitions/ashrae-energy-prediction/data

## Project Purpose

This project demonstrates how machine learning, anomaly detection, energy consumption analysis, and an interactive dashboard can support smarter electricity management on a university campus.

The simulated STIEI demonstration and the separate real-data experiment provide complementary ways to explore the system's capabilities and limitations.
