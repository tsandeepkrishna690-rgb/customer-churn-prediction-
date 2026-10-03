import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Remove unnecessary column
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Handle missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv(
    "data/telco_churn_cleaned.csv",
    index=False
)

print("Data cleaning completed successfully.")
print("Shape:", df.shape)
