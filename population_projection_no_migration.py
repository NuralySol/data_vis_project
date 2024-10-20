import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")

df_fertility = pd.read_csv("./fixtures/Formatted_Fertility_Rate_US.csv")
df_population = pd.read_csv("./fixtures/Formatted_Yearly_Population.csv")

df_fertility.columns = df_fertility.columns.str.strip()
df_population.columns = df_population.columns.str.strip()

df_combined = pd.merge(df_fertility, df_population, left_on="date", right_on="year")

X_population = df_combined[["year"]]  
y_population = df_combined["total_population"]  

linreg_population = LinearRegression()
linreg_population.fit(X_population, y_population)

future_years = np.array([[year] for year in range(2025, 2041)])
predicted_population = linreg_population.predict(future_years)

X_fertility = df_combined[["year"]]  
y_fertility = df_combined["Births_per_Woman"]  


linreg_fertility = LinearRegression()
linreg_fertility.fit(X_fertility, y_fertility)


predicted_fertility = linreg_fertility.predict(future_years)

# Historical mean of the fertility rate (Birth per woman). 
fertility_adjustment_factor = (
    predicted_fertility / df_combined["Births_per_Woman"].mean()
)  

adjusted_population = predicted_population * fertility_adjustment_factor

starting_population_2020 = 333  
actual_population_2020 = df_combined[df_combined["year"] == 2020][
    "total_population"
].values[0]

adjustment_factor_start = starting_population_2020 / actual_population_2020

adjusted_population *= adjustment_factor_start
predicted_population *= adjustment_factor_start

# Creating different scenarios for low, med, and high using the UN standard
low_growth_fertility = predicted_fertility * 0.95  
high_growth_fertility = predicted_fertility * 1.05  

low_adjusted_population = (
    predicted_population
    * (low_growth_fertility / df_combined["Births_per_Woman"].mean())
    * adjustment_factor_start
)
high_adjusted_population = (
    predicted_population
    * (high_growth_fertility / df_combined["Births_per_Woman"].mean())
    * adjustment_factor_start
)

#! Adding a starting point 333 million people for the year of 2020
years_with_2020 = np.insert(future_years.flatten(), 0, 2020)
predicted_population_with_2020 = np.insert(
    predicted_population, 0, starting_population_2020
)
adjusted_population_with_2020 = np.insert(
    adjusted_population, 0, starting_population_2020
)
low_adjusted_population_with_2020 = np.insert(
    low_adjusted_population, 0, starting_population_2020
)
high_adjusted_population_with_2020 = np.insert(
    high_adjusted_population, 0, starting_population_2020
)

plt.figure(figsize=(12, 8))

#! No fertility adjustment prediction of the population. 
sns.lineplot(
    x=years_with_2020,
    y=predicted_population_with_2020,
    label="Predicted Population (No Fertility Adj.)",
    color="red",
    linestyle="--",
    linewidth=2,
)

sns.lineplot(
    x=years_with_2020,
    y=adjusted_population_with_2020,
    label="Medium Growth Scenario (Fertility Adj.)",
    color="green",
    marker="o",
    linewidth=2,
    markersize=8,
)

sns.lineplot(
    x=years_with_2020,
    y=low_adjusted_population_with_2020,
    label="Low Growth Scenario (Fertility Adj.)",
    color="blue",
    marker="o",
    linewidth=2,
    markersize=8,
)

sns.lineplot(
    x=years_with_2020,
    y=high_adjusted_population_with_2020,
    label="High Growth Scenario (Fertility Adj.)",
    color="orange",
    marker="o",
    linewidth=2,
    markersize=8,
)

# Annotations for all of the scenarios. 
plt.annotate(
    f"2040 Projection (Medium): {adjusted_population_with_2020[-1]:.2f} million",
    xy=(2040, adjusted_population_with_2020[-1]),
    xytext=(2035, adjusted_population_with_2020[-1] + 10),
    arrowprops=dict(facecolor="orange", shrink=0.05, width=1.5, headwidth=8),
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="orange", boxstyle="round,pad=0.5"),
)

plt.annotate(
    f"2040 Projection (Low): {low_adjusted_population_with_2020[-1]:.2f} million",
    xy=(2040, low_adjusted_population_with_2020[-1]),
    xytext=(2035, low_adjusted_population_with_2020[-1] - 10),
    arrowprops=dict(facecolor="blue", shrink=0.05, width=1.5, headwidth=8),
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.5"),
)

plt.annotate(
    f"2040 Projection (High): {high_adjusted_population_with_2020[-1]:.2f} million",
    xy=(2040, high_adjusted_population_with_2020[-1]),
    xytext=(2035, high_adjusted_population_with_2020[-1] + 15),
    arrowprops=dict(facecolor="orange", shrink=0.05, width=1.5, headwidth=8),
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="orange", boxstyle="round,pad=0.5"),
)

plt.title(
    "US Population Prediction (Low, Medium, and High Growth Scenarios)",
    fontsize=16,
    weight="bold",
)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Total Population (Millions)", fontsize=14)
plt.xticks(np.arange(2020, 2041, 2), rotation=45)
plt.yticks(
    np.arange(
        min(low_adjusted_population_with_2020) - 10,
        max(high_adjusted_population_with_2020) + 20,
        20,
    )
)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.show()

# Print for the adjusted scenarions console output!
print(
    f"The adjusted population for 2040 (medium growth): {adjusted_population_with_2020[-1]:.2f} million"
)
print(
    f"The adjusted population for 2040 (low growth): {low_adjusted_population_with_2020[-1]:.2f} million"
)
print(
    f"The adjusted population for 2040 (high growth): {high_adjusted_population_with_2020[-1]:.2f} million"
)

#! Coefficient of dependency of population on fertility rate
coef_dependency = linreg_fertility.coef_[0]
print(
    f"The coefficient of dependency of population on fertility rate: {coef_dependency:.4f}"
)

# 	•	The coefficient of dependency of population on fertility rate is -0.0158.
# 	•	This means that for every one-unit decrease in the fertility rate (e.g., if the number of births per woman decreases by 1), the population is expected to decrease by 0.0158 million people (15,800 people), assuming all other factors remain constant.
# 	•	Since the coefficient is negative, it indicates an inverse relationship between the fertility rate and the population: as the fertility rate decreases, the population growth also decreases.
