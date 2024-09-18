
import seaborn as sns
import matplotlib.pyplot as plt
import umap # (Uniform Manifold Approximation and Projection) 2D

def visualize_clusters(data, clusters, method="K-means", ):
    
    #Redukcja wymiarów do 2D
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data)

    #Wykres punktowy:

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette="viridis", s=100, alpha=0.8)
    plt.title(f'Wizualizacja klastrów - {method}', fontsize=16)
    plt.xlabel('Wymiar 1')
    plt.ylabel('Wymiar 2')
    plt.legend(title='Klastry', loc='best')
    plt.show()





