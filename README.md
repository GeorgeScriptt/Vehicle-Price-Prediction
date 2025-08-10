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

* name: The full name of the vehicle, including make, model, and trim.
* description: A brief description of the vehicle, often including key features and selling points.
* make: The manufacturer of the vehicle (e.g., Ford, Toyota, BMW).
* model: The model name of the vehicle.
* year: The year the vehicle was manufactured.
* price: The price of the vehicle in USD.
* engine: Details about the engine, including type and specifications.
* cylinders: The number of cylinders in the vehicle's engine.
* fuel: The type of fuel used by the vehicle (e.g., Gasoline, Diesel, Electric).
* mileage: The mileage of the vehicle, typically in miles.
* transmission: The type of transmission (e.g., Automatic, Manual).
* trim: The trim level of the vehicle, indicating different feature sets or packages.
* body: The body style of the vehicle (e.g., SUV, Sedan, Pickup Truck)
* doors: The number of doors on the vehicle.
* exterior_color: The exterior color of the vehicle.
* interior_color: The interior color of the vehicle.
* drivetrain: The drivetrain of the vehicle (e.g., All-wheel Drive, Front-wheel Drive

# Descriptive Statistics of the Dataset
## Numerical Features
* year: Most vehicles are from 2024; small spread between 2023–2025.
* price: Ranges from $0 to $195,895 (mean ≈ $50,203); outliers and invalid zero values present.
* cylinders: Typically 4 or 6; some records contain 0 cylinders (likely data errors).
* mileage: Skewed distribution; many low-mileage entries consistent with new or nearly new cars.
* doors: Most vehicles have 4 doors, with very few 2- or 5-door models.

## Categorical Features
* make & model: 28 makes and 153 models represented; "Jeep" and "Hornet" are most common.
* body: Dominated by SUVs (70% of listings).
* fuel: Mostly gasoline-powered vehicles (67%).
* transmission: Predominantly automatic, especially 8-speed automatic.
* drivetrain: All-wheel drive is the most common configuration.
* color: High variety in both exterior (263 unique) and interior (91 unique) colors; "Bright White Clearcoat" and "Black" are most frequent.

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
* Price Distribution: The price distribution was skewed towards the lower end, with a long tail indicating the presence of high-priced luxury vehicles.
  [Price Distribution]("C:\Users\George\Documents\Personal Projects\Gradient Descent Algorithm\price_distribution.png")
