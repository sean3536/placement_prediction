# =========================
# Placement Prediction Project
# First EDA Pipeline
# =========================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Dataset
# =========================

df = pd.read_csv("Placement_Data_Full_Class.csv")

# =========================
# Basic Data Inspection
# =========================

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA INFO ==========")
print(df.info())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

# =========================
# Missing Values
# =========================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =========================
# Duplicate Rows
# =========================

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

# =========================
# Target Variable Distribution
# =========================

plt.figure(figsize=(6,4))

sns.countplot(x="status", data=df)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Count")

plt.show()

# =========================
# Gender vs Placement
# =========================

plt.figure(figsize=(6,4))

sns.countplot(x="gender", hue="status", data=df)

plt.title("Gender vs Placement Status")

plt.show()

# =========================
# Work Experience vs Placement
# =========================

plt.figure(figsize=(6,4))

sns.countplot(x="workex", hue="status", data=df)

plt.title("Work Experience vs Placement Status")

plt.show()

# =========================
# Degree Percentage vs Placement
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(x="status", y="degree_p", data=df)

plt.title("Degree Percentage vs Placement")

plt.show()

# =========================
# MBA Percentage vs Placement
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(x="status", y="mba_p", data=df)

plt.title("MBA Percentage vs Placement")

plt.show()

# =========================
# Correlation Heatmap
# =========================

# Select numeric columns only
numeric_df = df.select_dtypes(include=['number'])

# Correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

print("\n========== EDA COMPLETE ==========")