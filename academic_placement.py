import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Placement_Data_Cleaned (1).csv")

#print(df.head())
#print(df.info())
#print(df.describe())

# how many students were Placed vs Not Placed
df["status"].value_counts().plot(kind="bar")

plt.title("Placement Status Count")
plt.xlabel("Status")
plt.ylabel("Number of Students")
plt.show()

# boxplots to compare scores between placed and not placed students
# These plots help you see things like:
# Do placed students tend to have higher GPA/percentage scores?
# Are entrance test scores different between placed and not placed students?
# Does MBA percentage seem useful?

numeric_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

for col in numeric_cols:
    df.boxplot(column=col, by="status")
    plt.title(f"{col} by Placement Status")
    plt.suptitle("")
    plt.xlabel("Status")
    plt.ylabel(col)
    plt.show()


# Compare placement by category columns
categorical_cols = ["gender", "hsc_s", "degree_t", "workex", "specialisation"]

for col in categorical_cols:
    pd.crosstab(df[col], df["status"]).plot(kind="bar")
    plt.title(f"Placement Status by {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45)
    plt.show()

# Better version: placement rate by category
    # counts can be misleading

for col in categorical_cols:
    placement_rate = df.groupby(col)["status"].apply(lambda x: (x == "Placed").mean() * 100)

    placement_rate.plot(kind="bar")
    plt.title(f"Placement Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Placement Rate (%)")
    plt.xticks(rotation=45)
    plt.show()

# Correlation heatmap for numeric columns
numeric_df = df[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]]

corr = numeric_df.corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()