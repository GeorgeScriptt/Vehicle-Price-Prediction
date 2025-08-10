# Project Overview
This project aims to analyze a dataset of vehicle prices, build a regression model to predict price based on vehicle features, and present the findings. The process involves in-depth data exploration, feature engineering, training and evaluating multiple models, hyperparameter tuning, error analysis, and model interpretability.

# Libraries Used
* Pandas - Data Manipulation.
* Matplotlib - Data Visualization.
* Seaborn - Data Visualization.
* Scikit-Learn - Data Preprocessing, Machine Learning Models, Model Evaluation and Tuning.

# Dataset
The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/khwaishsaxena/vehicle-price-prediction-dataset/data). The dataset consists of 17 columns and 1002 rows.

Below is the description of each column in the dataset

* Name: The full name of the vehicle, including make, model, and trim.
* Description: A brief description of the vehicle, often including key features and selling points.
* Make: The manufacturer of the vehicle (e.g., Ford, Toyota, BMW).
* Model: The model name of the vehicle.
* Year: The year the vehicle was manufactured.
* Price: The price of the vehicle in USD.
* Engine: Details about the engine, including type and specifications.
* Cylinders: The number of cylinders in the vehicle's engine.
* Fuel: The type of fuel used by the vehicle (e.g., Gasoline, Diesel, Electric).
* Mileage: The mileage of the vehicle, typically in miles.
* Transmission: The type of transmission (e.g., Automatic, Manual).
* Trim: The trim level of the vehicle, indicating different feature sets or packages.
* Body: The body style of the vehicle (e.g., SUV, Sedan, Pickup Truck)
* Doors: The number of doors on the vehicle.
* Exterior_color: The exterior color of the vehicle.
* Interior_color: The interior color of the vehicle.
* Drivetrain: The drivetrain of the vehicle (e.g., All-wheel Drive, Front-wheel Drive

# Descriptive Statistics of the Dataset
## Numerical Features
* Year: Most vehicles are from 2024; small spread between 2023–2025.
* Price: Ranges from $0 to $195,895 (mean ≈ $50,203); outliers and invalid zero values present.
* Cylinders: Typically 4 or 6; some records contain 0 cylinders (likely data errors).
* Mileage: Skewed distribution; many low-mileage entries consistent with new or nearly new cars.
* Doors: Most vehicles have 4 doors, with very few 2- or 5-door models.

## Categorical Features
* Make & Model: 28 makes and 153 models represented; "Jeep" and "Hornet" are most common.
* Body: Dominated by SUVs (70% of listings).
* Fuel: Mostly gasoline-powered vehicles (67%).
* Transmission: Predominantly automatic, especially 8-speed automatic.
* Drivetrain: All-wheel drive is the most common configuration.
* Color: High variety in both exterior (263 unique) and interior (91 unique) colors; "Bright White Clearcoat" and "Black" are most frequent.

# Data Cleaning
A thorough data cleaning process was performed to ensure data quality and consistency before analysis. The key steps are outlined below:
## Duplicate Removal
* 24 duplicate rows were identified and removed based on identical values across all columns.
* This reduced the dataset size from 1,002 to 978 unique entries.
## Handling Missing Values
* Cylinders column had approximately 10% missing values, the highest among all features.
* All other columns with missing data had 5% and less missing values.
  
Missing values in numeric columns were imputed using the mean of each of the columns. This approach maintains the overall distribution while preventing data loss. Missing values in categorical columns were imputed with the mode(most frequent) of each of the column which helps preserve the dominant class in each category.

# Data Exploration Insights
* Price Distribution: The price distribution was skewed towards the lower end, with a long tail indicating the presence of high-priced luxury vehicles.![Price distribution](https://raw.githubusercontent.com/GeorgeScriptt/Vehicle-Price-Prediction/main/images/price_distribution.png)
* Mileage and Cylinders: Mileage also showed outliers and was heavily skewed towards lower values, which is expected for newer vehicles. The number of cylinders had a less skewed distribution but also showed some outliers.

## Relationship with Price
* Price and Fuel: Electric and Diesel vehicles tend to have higher price ranges, with Electric showing the most variability and outliers. Gasoline vehicles have a wide price spread with many outliers. Hybrid and PHEV vehicles fall in the mid-price range, while E85 Flex Fuel vehicles show consistent pricing. Diesel (B20 capable) has a single price point, indicating limited data.
![Price vs Fuel](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/price%20vs%20fuel.png?raw=true)
* Price and Make: RAM and Jeep have the highest median prices and widest ranges, indicating a variety of high-end models. Nissan, Volkswagen, and Kia generally offer more affordable options with lower medians. Mazda shows consistent mid-to-high pricing with little variation. Ford and Chevrolet display a wide spread, suggesting diverse pricing across their vehicle lineups.
![Price vs Make](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/price_make.png?raw=true)
* Price and Trim: Base, Tradesman, and Premium trims show the highest median prices with broader price ranges, reflecting more high-end features. Limited also displays wide variation, indicating diverse pricing within that trim. Laredo and Pursuit have moderate prices with tighter distributions. Latitude, 1.5T SE, SEL, and GT trims generally have lower to mid-range pricing, with Latitude being the most affordable overall.
![Price vs Trim](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/price%20vs%20trim.png?raw=true)
* Price and Mileage: Most data points are clustered near very low mileage, indicating a predominance of new or nearly-new vehicles in the dataset. There's a slight downward trend, suggesting that higher mileage may be associated with lower prices, though this relationship is not strongly linear. A few high-priced vehicles with both low and high mileage act as outliers, potentially representing luxury or rare models.
![Price vs Mileage](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/price%20vs%20mileage.png?raw=true)

## Feature Importance Based on F_Statistics
* Drivetrain, model, and fuel are the most influential features, with drivetrain showing the highest statistical impact on price.
* Make and trim also significantly affect pricing but to a lesser extent.
* Features like doors, exterior_color, and interior_color have minimal impact on price prediction.
This helps prioritize which features are most valuable when building predictive models.
![Feature Importance](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/feature_importance_ANOVA.png?raw=true)

## Correlation Heatmap
* Price vs Cylinders: Moderate positive correlation (0.43) – higher cylinder count is generally associated with higher prices.
* Price vs Mileage: Very weak positive correlation (0.08) – almost negligible relationship.
* Price vs Year: Near zero (0.00) – model year has no clear linear effect on price in this dataset.
![Correlation Heatmap](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/correlation_heatmap.png?raw=True)

# Feature Engineering
Based on the data exploration, feature engineering was performed to enhance the model's predictive capability:
* Age: Vehicle age was derived from subtracting the year from the current year from python's datetime module.

The engineered feature was added to the dataset alongside the original features (excluding 'name', 'description', 'engine' , and 'year' column which were dropped due to irrelevance or redundancy after extraction).

# Model Exploration and Comparison
Several regression models were explored to identify the most suitable one for this task:
1. Linear Rgression
2. Ridge Regression
3. Stochastic Gradient Descent Regression

The initial performance of these models on the test set (before hyperparameter tuning) was as follows:
1. Linear Regression: MSE = 82135250.0, R2= 0.83
2. Ridge: MSE = 82512800.0, R2 = 0.83
3. SGDRgressor: MSE = 81719460.0, R2 = 0.83

SGDRegressor Model with parameter `max_iter=6000` (number of iterations) slightly outperforms the other models, but differences are minimal.

# Hyperparameter Tuning and Cross-Validation
Hyperparameter tuning was performed for Ridge and SGDRgressor model, using 5-fold cross-validation with GridSearchCV to find the optimal model configurations and get a more robust performance estimate.
* Ridge Tuning: The best hyperparameter found for Ridge was `alpha = 1.0`.
* Random Forest Tuning: The best hyperparameters found for Random Forest were `'alpha' = 0.0001, 'eta0' = 0.1, 'learning_rate' = 'adaptive', 'loss' = 'squared_error', 'max_iter' = 8000, 'penalty' = 'l2'`.

The performance of the tuned models on the test set was as follows:
* Tuned Ridge: Test MSE = 78452834.45, R2 = 0.84
* Tuned SGDRegressor: Test MSE = 82147742.43, R2 = 0.83

The tuned Ridge model had a `1%` increase in R2 score.

# Best Model Selection
Based on the evaluation metrics after hyperparameter tuning, the **Tuned Ridge Regression** model was selected as the best-performing model. It achieved the lowest Mean Squared Error (78452834.45) and the highest R-squared score (0.84) on the test set, indicating the best balance between minimizing prediction errors and explaining the variance in vehicle prices.

# Error Analysis
* Actual vs Predicted Price Plot: The plot showed a strong linear relationship between actual and predicted prices, but with increasing scatter as the price increased, indicating higher absolute errors for more expensive vehicles.
![Actual vs Predicted](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/actual%20vs%20predicted%20price%20best_model.png?raw=True)
* Distribution Plot of Actual and Predicted Price: The kernel density estimates curve for actual price (red) and predicted price (blue) helps visualize how well the predicted values align with the real data. The close overlap of the two curves indicates that the model captures the overall price distribution well, suggesting good performance in modeling price behavior. Both curves peak at around 45,000–50,000, showing that the model correctly identifies the most common price range. However, the predicted peak is slightly higher, suggesting that the model may slightly overestimate the density in this price range. In the higher price range (above 100,000), the predicted curve is slightly lower than the actual curve, implying that the model underestimates the density of high-value items.

These findings suggest that the model's performance is less consistent at the extreme ends of the price spectrum

# Model Interpretability
Analyzing the coefficients of the tuned Ridge model provides insights into which features have the largest impact on the predicted price
* Features with Largest Positive Impact: Specific high-end models (e.g., 'Grand Wagoneer', 'i7') and makes ('BMW') are associated with the largest increases in predicted price. Premium interior color ('Caramel') also show significant positive coefficients, likely correlating with luxury trims and features.
* Features with Largest Negative Impact: Certain standard or lower-priced models (e.g., 'SQ5' 'Compass', 'Mustang Mach-E') and makes ('Nissan', and 'Kia') are associated with the largest decreases in predicted price. Specific utilitarian trims ('Tradesman Regular Cab 4x4 8' Box') also have a strong negative impact.
* The coefficients indicate the estimated change in price associated with a one-unit change in a numerical feature or the presence of a specific category (for one-hot encoded features), holding other features constant. This provides a clear understanding of the relative importance and direction of influence of different vehicle characteristics on price according to the model.

# Conclusion
The Ridge Regression model with parameter `alpha = 0.1` provides a reasonably good prediction of vehicle prices based on the available features, achieving an R-squared of 0.84. The analysis highlighted the importance of make, model, trim, and cylinders in determining price. Error analysis revealed that the model's accuracy decreases for very high-priced vehicles
