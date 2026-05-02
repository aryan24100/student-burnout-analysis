# ======================================================
# STUDENT BURNOUT DATA ANALYSIS PROJECT
# ======================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ======================================================
# Step 2: Load Dataset
# ======================================================

df = pd.read_csv("data/student_burnout_dataset.csv")

print("Dataset Loaded Successfully!\n")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# ======================================================
# Step 3: Classify Columns (Numerical & Categorical)
# ======================================================

print("\nColumn Classification:")

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\nNumerical Columns:")
print(list(numerical_cols))

print("\nCategorical Columns:")
print(list(categorical_cols))

# ======================================================
# Step 4: Data Cleaning
# ======================================================

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ======================================================
# Step 5: Statistical Analysis
# ======================================================

print("\n--- MEAN ---")
print(df[numerical_cols].mean())

print("\n--- MEDIAN ---")
print(df[numerical_cols].median())

print("\n--- MODE ---")
print(df.mode().iloc[0])

# ======================================================
# Step 6: Burnout Distribution
# ======================================================

print("\nBurnout Level Distribution:")
print(df['burnout_level'].describe())

# ======================================================
# Step 7: Correlation Analysis
# ======================================================

print("\nCorrelation Matrix:")
print(df[numerical_cols].corr())

# ======================================================
# Step 8: Graphical Analysis
# ======================================================

# 1️⃣ Sleep vs Burnout
plt.figure()
plt.scatter(df['sleep_hours'], df['burnout_level'])
plt.xlabel("Sleep Hours")
plt.ylabel("Burnout Level")
plt.title("Sleep Hours vs Burnout Level")
plt.show()

# 2️⃣ Screen Time vs Burnout
plt.figure()
plt.scatter(df['screen_time_hours'], df['burnout_level'])
plt.xlabel("Screen Time Hours")
plt.ylabel("Burnout Level")
plt.title("Screen Time vs Burnout Level")
plt.show()

# 3️⃣ Productivity vs Exam Score
plt.figure()
plt.scatter(df['productivity_score'], df['exam_score'])
plt.xlabel("Productivity Score")
plt.ylabel("Exam Score")
plt.title("Productivity vs Exam Score")
plt.show()

# 4️⃣ Burnout Histogram
plt.figure()
plt.hist(df['burnout_level'], bins=20)
plt.xlabel("Burnout Level")
plt.ylabel("Frequency")
plt.title("Burnout Level Distribution")
plt.show()

# ================================================-======
# Step 9: Final Cleaned Dataset Preview
# ======================================================

print("\nFinal Dataset Preview:")
print(df.head())

#ML Model
from sklearn.linear_model import LinearRegression

X = df[['sleep_hours','study_hours','screen_time_hours','focus_index','exercise_minutes','social_media_hours']]
y = df['burnout_level']

model = LinearRegression()
model.fit(X, y)

print("Model Trained and Ready!")