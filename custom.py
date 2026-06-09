import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 1. Load The Dataset
data=pd.read_csv('Mall_Customers.csv')
print("First 5 Entries:\n")
print(data.head())
print(data.isnull().sum())
# 2. Feature Selection
X=data[['Annual Income (k$)','Spending Score (1-100)']]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
# 3. Train K-Means Model
# Using k=5 as it fits this dataset
optimal_k=5
kmeans=KMeans(n_clusters=optimal_k,init='k-means++',random_state=42)
data['Cluster']=kmeans.fit_predict(X_scaled)
# Calculate centers dynamically to guarantee the correct labels every time
centers=data.groupby('Cluster')[['Annual Income (k$)','Spending Score (1-100)']].mean()
def assign_person(row):
    indx=row['Cluster']
    income=centers.loc[indx,'Annual Income (k$)']
    spend=centers.loc[indx,'Spending Score (1-100)']
    if income>70 and spend>60:
        return "High Income, High Spending(VIPs)"
    elif income>70 and spend<=40:
        return "High Income, Low Spending(Careful Buyers)"
    elif income<40 and spend>60:
        return "Low Income, High Spending(Impulsive Buyers)"
    elif income<40 and spend<=40:
        return "Low Income, Low Spending(Frugal Buyers)"
    else:
        return " Average Income, Average Spending(Standard)"
data['Segment_Name']=data.apply(assign_person,axis=1)
# 4. Data Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x='Annual Income (k$)',
                y='Spending Score (1-100)',
                edgecolor='black',
                hue='Segment_Name',
                palette='Set1',
                s=100,
                data=data)
plt.title('Customer Segments Using K-Means Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=11)
plt.ylabel('Spending Score (1-100)', fontsize=11)
plt.legend(title='Customer Segments', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# 5. Interpret Cluster Characteristics
print("\n Final Customer Segment Analysis (Averages)")
print(data.groupby('Segment_Name')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
