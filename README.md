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

## Corelation Heatmap
* Price vs Cylinders: Moderate positive correlation (0.43) – higher cylinder count is generally associated with higher prices.
* Price vs Mileage: Very weak positive correlation (0.08) – almost negligible relationship.
* Price vs Year: Near zero (0.00) – model year has no clear linear effect on price in this dataset.
![Correlation Heatmap](https://github.com/GeorgeScriptt/Vehicle-Price-Prediction/blob/main/images/correlation_heatmap.png?raw=True)

# Feature Engineering
