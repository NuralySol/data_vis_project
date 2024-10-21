import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the population data
df_population = pd.read_csv("./fixtures/Formatted_Yearly_Population.csv")

# Display the first few rows to check the data
print(df_population.head())

# Convert 'total_population' to float
df_population["total_population"] = df_population["total_population"].astype(float)

# Split the data into train and test sets and Features
X = df_population[["year"]]
y = df_population["total_population"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the feature (year)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear regression model and fit
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = linreg.predict(X_test_scaled)

# Prediction up to year 2040
future_years = np.array([[year] for year in range(2021, 2041)])
future_years_scaled = scaler.transform(future_years)
future_population_pred = linreg.predict(future_years_scaled)

# Plot the actual data, test predictions, and future projections
plt.figure(figsize=(12, 8))

# Scatter plot for the actual data
plt.scatter(
    df_population["year"],
    df_population["total_population"],
    color="blue",
    label="Actual Population",
    s=80,  # Increase point size for better visibility
    alpha=0.7
)

# Plot predicted population from the test set
plt.plot(
    X_test, y_pred, color="red", linewidth=2, label="Predicted Population (Test Set)"
)

# Plot future population predictions up to 2040
plt.plot(
    future_years,
    future_population_pred,
    "g--",
    label="Predicted Population (2040)",
    marker="o",
    markersize=8
)

# Annotate the 2040 projection, shifted down with an arrow pointing to the actual point
plt.annotate(
    f"2040 Projection: {future_population_pred[-1]:.2f} million",
    xy=(2040, future_population_pred[-1]),  # Pointing to the actual data point
    xytext=(2032, future_population_pred[-1] - 15),  # Shifted annotation downward
    arrowprops=dict(facecolor="orange", arrowstyle="->", lw=2),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="orange", lw=1.5)
)

# Title and labels
plt.title("Population Prediction Up to 2040", fontsize=18, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Total Population (Millions)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# Display the current and predicted population
current_population = df_population["total_population"].iloc[-1]
print(
    f"The current population (most recent data point) is: {current_population:.2f} million"
)

# Display the 2040 projection
predicted_population_2040 = future_population_pred[-1]
print(
    f"The predicted population for the year 2040 is: {predicted_population_2040:.2f} million"
)