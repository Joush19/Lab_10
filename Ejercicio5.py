import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

data = {
    'Avg Purchase Cost (Bs)': [750, 1245, 230, 533, 490, 1000, 190, 900, 600, 50, 1100, 930, 450, 330, 750],
    'Avg Purchases with Credit Card': [3, 1, 4, 3, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 0]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 10), inertia, marker='o', linestyle='--')
plt.title('Método del Codo para determinar el número de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print(df)

plt.figure(figsize=(8, 6))
plt.scatter(df['Avg Purchase Cost (Bs)'], df['Avg Purchases with Credit Card'], c=df['Cluster'], cmap='viridis')
plt.title('Clusters de clientes')
plt.xlabel('Costo Promedio de Compra (Bs)')
plt.ylabel('Número Promedio de Compras con Tarjeta')
plt.colorbar(label='Cluster')
plt.show()

for k in [2, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f'Cluster_{k}'] = kmeans.fit_predict(scaled_data)
    print(f"\nClusters con {k} clusters:")
    print(df[[f'Avg Purchase Cost (Bs)', f'Avg Purchases with Credit Card', f'Cluster_{k}']])
