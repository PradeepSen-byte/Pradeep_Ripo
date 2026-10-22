# Customer Segmentation using K-Means Clustering

This project implements customer segmentation using the **K-Means Clustering** unsupervised learning algorithm on the Mall Customers dataset (`Mall_Customers.csv`).

## 📌 Project Overview
Customer segmentation helps businesses identify distinct groups of customers based on their annual income and spending score, enabling targeted marketing strategies and improved customer engagement.

## 🚀 Key Steps in the Project
1. **Data Understanding & Exploration**:
   - Inspecting summary statistics, distributions of Age, Annual Income, and Spending Score.
   - Gender count visualization.
2. **Data Preprocessing**:
   - Dropping unnecessary identifiers (`CustomerID`).
   - Categorical encoding for gender (`LabelEncoder`).
   - Feature standardisation using `StandardScaler`.
3. **Determining Optimal Number of Clusters (K)**:
   - **Elbow Method**: Evaluating Within-Cluster Sum of Squares (WCSS) across cluster counts ($K=1$ to $10$).
   - **Silhouette Analysis**: Measuring separation distance between resulting clusters.
4. **Model Training & Evaluation**:
   - Applying `KMeans(n_clusters=5)`.
   - Cluster label assignment.
5. **Visualization**:
   - Scatter plots of customer clusters with distinct colors.
   - Inverse transformation and plotting of cluster centroids.

## 🛠️ Requirements & Installation
Install the necessary Python packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 💻 How to Run
Run the Python script directly:
```bash
python kmeans_customer.py
```
Or explore interactively using the Jupyter notebook:
- `class_6.ipynb`
