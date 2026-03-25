1. Problem statement
The goal of this project is to predict the duration of  taxi trip in New York City using historical trip data.

2. Dataset
Source: NYC Taxi Trip Dataset
Total records:~1.45 million
Target Variable: trip_duration

3. Data Cleaning 
Removed invalid passenger counts like [0,7,8,9]
Filtered extreme trip durations (outliers) like in between 60 seconds to the 7200 seconds
Filtered the coordinates like latitude should be in 40 to 42 degrees() North to south  and longitude should be in -75 to -73 degrees east to west
Handling missing/null values 
Removed data leakage columns(dropoff_time)

4. Feature Engineering
Created meaningful features such as 
1. Time-based features: pickup_hours, pick_weekday, pickup_month, is_weekend, is_rush_hour, is_night
2. Geospatial Features: trip_distance_km, latitude_difference, longitude_difference, direction
3. encoded categorical variables using one-hot encoding: vendor_id_encoded, store_and_fwd_flag_encoded

5. Model used
used regression model because the target variable consist of real numbers 
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor
5. Tuned XGBoost

6. model Performance 

| Model             | RMSE | MAE | R²   |
| ----------------- | ---- | --- | ---- |
| Linear Regression | 403  | 277 | 0.62 |
| Decision Tree     | 378  | 254 | 0.66 |
| Random Forest     | 377  | 256 | 0.66 |
| XGBoost           | 315  | 206 | 0.76 |
| Tuned XGBoost     | 313  | 204 | 0.77 |

7. key Insights
The correlation heatmap shows that trip distance has the strongest linear relationship with trip duration (0.77). However, many time-based features such as pickup hour show low linear correlation. Despite this, these features are highly important in tree-based models like XGBoost, as they capture non-linear patterns such as traffic conditions and rush hour effects.
Non-linear models  outperform linear models
XGBoost significantly improves performance
Hyperparameter tuning provides marginal gains after strong feature engineering 

8. Limitations
Missing real-world features like: traffic conditions,weather,road conditions
Presence of irreducible error limits model performance

9. Future Improvements
Add real-time traffic data
Include weather features
Deploy model using API (FastAPI/Streamlit)





