# ==========================
# PART A : DATA UNDERSTANDING
# ==========================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Last 5 Rows
print("\nLast 5 Rows:")
print(df.tail())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Column Names
print("\nColumn Names:")
print(df.columns)

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# ==========================
# PART B : EDA
# ==========================

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.show()

# Male vs Female Count
plt.figure(figsize=(5,4))
sns.countplot(x="Gender", data=df)
plt.title("Male vs Female Customers")
plt.show()

# ==========================
# PART C : DATA PREPROCESSING
# ==========================

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop CustomerID
df = df.drop("CustomerID", axis=1)

# Convert Gender into numbers
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Select Features
X = df

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData after Preprocessing:")
print(df.head())

print("\nScaled Data Shape:")
print(X_scaled.shape)

# ==========================
# PART D : ELBOW METHOD
# ==========================

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# ==========================
# SILHOUETTE SCORE
# ==========================

from sklearn.metrics import silhouette_score

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K = {i}, Silhouette Score = {score:.3f}")

# ==========================
# PART E : K-MEANS MODEL
# ==========================

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Labels:")
print(df.head())

# ==========================
## ==========================
# PART F : CLUSTER VISUALIZATION
# ==========================

plt.figure(figsize=(8,6))

# Scatter Plot
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

# Centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:,2],     # Annual Income
    centers[:,3],     # Spending Score
    c="red",
    s=250,
    marker="X",
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments using K-Means")
plt.legend()

plt.show()