import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_fertility = pd.read_csv("./fixtures/Formatted_Fertility_Rate_US.csv")
df_fertility.columns = df_fertility.columns.str.strip()
df_fertility["year"] = pd.to_datetime(df_fertility["date"], format="%Y").dt.year


df_fertility = df_fertility[["year", "Births_per_Woman"]]  
X = df_fertility[["year"]]
y = df_fertility["Births_per_Woman"]

# Split the data into train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#! LinearRegression algorithm
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

y_pred = linreg.predict(X_test_scaled)

future_years = np.array([[year] for year in range(2021, 2041)])
future_years_scaled = scaler.transform(future_years)
future_fertility_pred = linreg.predict(future_years_scaled)

plt.figure(figsize=(12, 8))

plt.scatter(
    df_fertility["year"],
    df_fertility["Births_per_Woman"],
    color="blue",
    label="Actual Fertility Rate",
    s=100,
    alpha=0.6,
)

plt.plot(
    X_test,
    y_pred,
    color="red",
    linewidth=2,
    label="Predicted Fertility Rate (Test Set)",
    linestyle="--",
    alpha=0.7,
)

plt.plot(
    future_years,
    future_fertility_pred,
    "g--",
    label="Predicted Fertility Rate (2021-2040)",
    marker="D",
    markersize=8,
)

plt.fill_between(
    future_years.flatten(),
    future_fertility_pred - 0.05,
    future_fertility_pred + 0.05,
    color="green",
    alpha=0.1,
    label="Confidence Interval",
)


plt.annotate(
    f"2040 Projection: {future_fertility_pred[-1]:.2f} births/woman",
    xy=(2040, future_fertility_pred[-1]),  
    xytext=(2030, future_fertility_pred[-1] + 0.45),  
    arrowprops=dict(facecolor="orange", shrink=0.05, width=1, headwidth=10),
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="orange", boxstyle="round,pad=0.5"),
)

plt.title("Fertility Rate Prediction (Linear Regression)", fontsize=18, weight="bold")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Births per Woman", fontsize=14)
plt.xticks(np.arange(1960, 2045, step=5), rotation=45)
plt.yticks(np.arange(1.5, 4, step=0.2))
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

predicted_fertility_2040 = future_fertility_pred[-1]
print(
    f"The predicted fertility rate for the year 2040 is: {predicted_fertility_2040:.2f} births per woman"
)
