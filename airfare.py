import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generating Dummy Data
np.random.seed(42)

# Creating a synthetic dataset for airfare pricing
num_samples = 2000
data = pd.DataFrame({
    "flight_distance_km": np.random.randint(200, 10000, num_samples),
    "booking_time_days": np.random.randint(1, 365, num_samples),  # Days before flight booked
    "demand_factor": np.random.uniform(0.5, 2.0, num_samples),  # Market demand multiplier
    "seat_availability": np.random.uniform(0.2, 1.0, num_samples),  # 0.2 means 20% seats left
    "seasonality_factor": np.random.choice([0.8, 1.0, 1.2], num_samples),  # Off-peak, normal, peak
    "fare": np.random.randint(50, 1500, num_samples)  # Airfare in USD
})

# Splitting into features and target
X = data.drop(columns=["fare"])
y = data["fare"]

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Gradient Boosting Model for Airfare Prediction
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predicting on test set
y_pred = gb_model.predict(X_test)

# Visualizing Predictions vs. Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Predicted vs. Actual Airfare")
plt.show()

# Creating a synthetic dataset for traveler segmentation
traveler_data = pd.DataFrame({
    "avg_spend_per_trip": np.random.randint(100, 5000, num_samples),
    "num_flights_per_year": np.random.randint(1, 20, num_samples),
    "loyalty_points": np.random.randint(0, 50000, num_samples),
    "business_travel_ratio": np.random.uniform(0, 1, num_samples),  # Ratio of business trips
})

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(traveler_data)

# Applying K-Means Clustering
num_clusters = 4  # Assuming 4 types of travelers
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
traveler_data["cluster"] = kmeans.fit_predict(scaled_data)

# Visualizing Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=traveler_data["avg_spend_per_trip"], y=traveler_data["num_flights_per_year"], hue=traveler_data["cluster"], palette="viridis")
plt.xlabel("Avg Spend per Trip (USD)")
plt.ylabel("Flights per Year")
plt.title("Traveler Segmentation Clusters")
plt.legend(title="Cluster")
plt.show()

# Saving results to CSV
data.to_csv("airfare_pricing_data.csv", index=False)
traveler_data.to_csv("traveler_segmentation_data.csv", index=False)
