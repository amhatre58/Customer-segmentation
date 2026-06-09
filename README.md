## Customer Segmentation Using Unsupervised Learning
This project applies machine learning clustering techniques to group customers based on their purchasing behavior, income levels, and spending patterns. By leveraging unsupervised learning, businesses can uncover hidden structures in customer data to design targeted marketing strategies and improve customer retention.
------------------------------
## 📌 Project Overview
Generic marketing campaigns are often inefficient and expensive. This project solves that problem by implementing K-Means Clustering to divide a customer base into distinct, actionable segments. Each segment represents a unique consumer profile, allowing businesses to personalize product recommendations and optimize budget allocation.
------------------------------
## ⚙️ Core Workflow## 1. Data Preprocessing & Feature Engineering

* Handling Outliers: Identifying and managing extreme income or spending values that could distort cluster boundaries.
* Feature Selection: Isolating key behavioral drivers such as Annual Income, Spending Score, and Purchase Frequency.
* Feature Scaling: Applying StandardScaler or MinMaxScaler to ensure all numerical features contribute equally to distance calculations.

## 2. Determining Optimal Clusters

* The Elbow Method: Plotting Within-Cluster Sum of Squares (WCSS) against the number of clusters to find the "elbow point."
* Silhouette Analysis: Computing silhouette scores to evaluate how well-separated and cohesive the resulting clusters are.

## 3. Clustering & Visualization

* Model Fitting: Training the K-Means algorithm on the scaled behavioral data.
* Dimensionality Reduction: Utilizing Principal Component Analysis (PCA) or t-SNE to compress features into 2D or 3D space for easier plotting.
* Cluster Interpretation: Creating scatter plots, box plots, and radar charts to visualize and name each customer segment (e.g., "High Income, Low Spenders").

------------------------------
## 📊 Business Insights & Applications
The final output provides clear, data-driven profiles used by marketing teams for:

* High-Value Customers: Enrolling them into exclusive loyalty programs.
* Price-Sensitive Shoppers: Targetting them with strategic discount coupons and bundle offers.
* At-Risk Segments: Launching re-engagement email campaigns to prevent churn.

