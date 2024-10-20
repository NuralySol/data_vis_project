import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk")

df_fertility = pd.read_csv("./fixtures/Formatted_Fertility_Rate_US.csv")
df_population = pd.read_csv("./fixtures/Formatted_Yearly_Population.csv")
df_migration = pd.read_csv("./fixtures/Formatted_Migration_Rate_US.csv")

df_fertility.columns = df_fertility.columns.str.strip()
df_population.columns = df_population.columns.str.strip()
df_migration.columns = df_migration.columns.str.strip()

df_combined = pd.merge(
    pd.merge(df_fertility, df_population, left_on="date", right_on="year"),
    df_migration,
    on="year",
)

X = df_combined[["year"]] 
y_population = df_combined["total_population"]  
y_fertility = df_combined["Births_per_Woman"]  
y_migration = df_combined["Net_Migration_Rate"]  

#! LinearRegression on populatio, fertility, and migration to the US
linreg_population = LinearRegression()
linreg_fertility = LinearRegression()
linreg_migration = LinearRegression()

linreg_population.fit(X, y_population)
linreg_fertility.fit(X, y_fertility)
linreg_migration.fit(X, y_migration)

future_years = np.array([[year] for year in range(2025, 2041)])
predicted_population = linreg_population.predict(future_years)
predicted_fertility = linreg_fertility.predict(future_years)
predicted_migration = linreg_migration.predict(future_years)

fertility_adjustment_factor = (
    predicted_fertility / df_combined["Births_per_Woman"].mean()
)
migration_adjustment_factor = (
    predicted_migration / df_combined["Net_Migration_Rate"].mean()
)

adjusted_population = (
    predicted_population * fertility_adjustment_factor * migration_adjustment_factor
)

starting_population_2020 = 333  
actual_population_2020 = df_combined[df_combined["year"] == 2020][
    "total_population"
].values[0]

adjustment_factor_start = starting_population_2020 / actual_population_2020

adjusted_population *= adjustment_factor_start
predicted_population *= adjustment_factor_start

#! Different scenarios 
low_growth_fertility = predicted_fertility * 0.95  
high_growth_fertility = predicted_fertility * 1.05 
low_growth_migration = predicted_migration * 0.95  
high_growth_migration = predicted_migration * 1.05  

low_adjusted_population = (
    predicted_population
    * (low_growth_fertility / df_combined["Births_per_Woman"].mean())
    * (low_growth_migration / df_combined["Net_Migration_Rate"].mean())
    * adjustment_factor_start
)
high_adjusted_population = (
    predicted_population
    * (high_growth_fertility / df_combined["Births_per_Woman"].mean())
    * (high_growth_migration / df_combined["Net_Migration_Rate"].mean())
    * adjustment_factor_start
)

#! Starting population 333 million from the year of 2020
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
sns.lineplot(
    x=years_with_2020,
    y=predicted_population_with_2020,
    label="Predicted Population (No Adjustments)",
    color="red",
    linestyle="--",
    linewidth=2,
)

sns.lineplot(
    x=years_with_2020,
    y=adjusted_population_with_2020,
    label="Medium Growth Scenario (Adjusted)",
    color="green",
    marker="o",
    linewidth=2,
    markersize=8,
)

sns.lineplot(
    x=years_with_2020,
    y=low_adjusted_population_with_2020,
    label="Low Growth Scenario (Adjusted)",
    color="blue",
    marker="o",
    linewidth=2,
    markersize=8,
)

sns.lineplot(
    x=years_with_2020,
    y=high_adjusted_population_with_2020,
    label="High Growth Scenario (Adjusted)",
    color="orange",
    marker="o",
    linewidth=2,
    markersize=8,
)

plt.annotate(
    f"Medium: {adjusted_population_with_2020[-1]:.2f}M",
    xy=(2040, adjusted_population_with_2020[-1]),
    xytext=(2037, adjusted_population_with_2020[-1] + 10),
    arrowprops=dict(facecolor="green", arrowstyle="->"),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green"),
)

plt.annotate(
    f"Low: {low_adjusted_population_with_2020[-1]:.2f}M",
    xy=(2040, low_adjusted_population_with_2020[-1]),
    xytext=(2037, low_adjusted_population_with_2020[-1] - 10),
    arrowprops=dict(facecolor="blue", arrowstyle="->"),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="blue"),
)

plt.annotate(
    f"High: {high_adjusted_population_with_2020[-1]:.2f}M",
    xy=(2040, high_adjusted_population_with_2020[-1]),
    xytext=(2037, high_adjusted_population_with_2020[-1] + 10),
    arrowprops=dict(facecolor="orange", arrowstyle="->"),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="orange"),
)

plt.title("US Population Prediction", fontsize=16, weight="bold")
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

print(
    f"The adjusted population for 2040 (medium growth): {adjusted_population_with_2020[-1]:.2f} million"
)
print(
    f"The adjusted population for 2040 (low growth): {low_adjusted_population_with_2020[-1]:.2f} million"
)
print(
    f"The adjusted population for 2040 (high growth): {high_adjusted_population_with_2020[-1]:.2f} million"
)

fertility_coefficient = linreg_fertility.coef_[0]
migration_coefficient = linreg_migration.coef_[0]

print(f"Coefficient of dependency on fertility rate: {fertility_coefficient:.4f}")
print(f"Coefficient of dependency on migration rate: {migration_coefficient:.4f}")

if abs(fertility_coefficient) > abs(migration_coefficient):
    print("Fertility rate has a larger impact on population growth than migration rate.")
else:
    print("Migration rate has a larger impact on population growth than fertility rate.")