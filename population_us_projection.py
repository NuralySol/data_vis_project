import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_population = pd.read_csv("./fixtures/Formatted_Yearly_Population.csv")

# first rows of the head to check the data
print(df_population.head())


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


y_pred = linreg.predict(X_test_scaled)

# Prediction up to year 2040
future_years = np.array([[year] for year in range(2021, 2041)])
future_years_scaled = scaler.transform(future_years)
future_population_pred = linreg.predict(future_years_scaled)

plt.figure(figsize=(10, 8))
plt.scatter(
    df_population["year"],
    df_population["total_population"],
    color="blue",
    label="Actual Population",
)

plt.plot(
    X_test, y_pred, color="red", linewidth=2, label="Predicted Population (Test Set)"
)

plt.plot(
    future_years,
    future_population_pred,
    "g--",
    label="Predicted Population (2040)",
    marker="o",
)
plt.annotate(
    "2040 Projection",
    xy=(2040, future_population_pred[-1]),
    xytext=(2035, future_population_pred[-1] + 2),
    arrowprops=dict(facecolor="orange", shrink=0.05, width=1, headwidth=10),
)

plt.title("Population Prediction Up to 2040")
plt.xlabel("Year")
plt.ylabel("Total Population (Millions)")
plt.legend()
plt.grid(True)
plt.show()
current_population = df_population["total_population"].iloc[-1]
print(
    f"The current population (most recent data point) is: {current_population:.2f} million"
)
predicted_population_2040 = future_population_pred[-1]
print(
    f"The predicted population for the year 2040 is: {predicted_population_2040:.2f} million"
)
