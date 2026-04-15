# ============================================================
# TASK 1 - BIG DATA ANALYSIS (SIMPLIFIED VERSION)
# CodTech Internship - Mallavarapu Venkata Sai
# ============================================================

import pandas as pd
import os
import pandas as pd

df = pd.read_csv("../data/yellow_tripdata_2016-02.csv", nrows=50000)
df.to_csv("../data/sample.csv", index=False)
# Paths
DATA_PATH = "../data/sample.csv"
OUTPUT_PATH = "../outputs/insights.txt"

# Create outputs folder if not exists
os.makedirs("../outputs", exist_ok=True)

print("=" * 50)
print("BIG DATA ANALYSIS - STARTED")
print("=" * 50)

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("Data Loaded Successfully!")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. Basic Information
# ------------------------------------------------------------
print("\nColumns in dataset:")
print(df.columns)

# ------------------------------------------------------------
# 3. Analysis 1: Passenger Distribution
# ------------------------------------------------------------
print("\nPassenger Count Distribution:")
passenger_dist = df["passenger_count"].value_counts()
print(passenger_dist)

# ------------------------------------------------------------
# 4. Analysis 2: Trip Distance
# ------------------------------------------------------------
avg_distance = df["trip_distance"].mean()
print(f"\nAverage Trip Distance: {avg_distance:.2f} miles")

# ------------------------------------------------------------
# 5. Analysis 3: High Fare Trips
# ------------------------------------------------------------
high_fare = df[df["fare_amount"] > 50]
print(f"\nHigh Fare Trips Count: {len(high_fare)}")

# ------------------------------------------------------------
# 6. Analysis 4: Payment Type Distribution
# ------------------------------------------------------------
payment_dist = df["payment_type"].value_counts()
print("\nPayment Type Distribution:")
print(payment_dist)

# ------------------------------------------------------------
# 7. Save Insights
# ------------------------------------------------------------
with open(OUTPUT_PATH, "w") as f:
    f.write("BIG DATA ANALYSIS INSIGHTS\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Total Rows: {len(df)}\n\n")

    f.write("Passenger Distribution:\n")
    f.write(str(passenger_dist) + "\n\n")

    f.write(f"Average Trip Distance: {avg_distance:.2f} miles\n\n")

    f.write(f"High Fare Trips (>50): {len(high_fare)}\n\n")

    f.write("Payment Type Distribution:\n")
    f.write(str(payment_dist) + "\n\n")

    f.write("Key Insights:\n")
    f.write("- Most trips have low passenger counts (1–2 people)\n")
    f.write("- Average trip distance is relatively short\n")
    f.write("- High fare trips are less frequent\n")
    f.write("- Certain payment types dominate usage\n")

print("\nInsights saved to outputs/insights.txt")

print("\n" + "=" * 50)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 50)