import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Clustering.csv')

print("Primeras filas del dataset:")
print(df.head())

X = df[['x', 'y']]  

print(f'\n¿Existen valores NaN en el dataset?: {X.isnull().any()}')

X = X.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []  
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Método del Codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

n_clusters_optimal = 3  

kmeans = KMeans(n_clusters=n_clusters_optimal, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

df['Cluster'] = y_kmeans

print("\nPrimeras filas con los clusters asignados:")
print(df.head())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans_pca = KMeans(n_clusters=n_clusters_optimal, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans_pca = kmeans_pca.fit_predict(X_pca)

df['PCA_Cluster'] = y_kmeans_pca

print("\nPrimeras filas con los clusters después de PCA:")
print(df.head())

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('K-Means en el espacio original')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['PCA_Cluster'], cmap='viridis')
plt.title('K-Means en el espacio reducido por PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()
