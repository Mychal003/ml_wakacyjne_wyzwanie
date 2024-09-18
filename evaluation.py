from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clusters(data, clusters, method="K-means"):
    if len(set(clusters)) > 1: 
        
        silhouette_avg = silhouette_score(data, clusters)
        print(f"{method} - Silhouette Score: {silhouette_avg:.4f}")

        davies_bouldin_avg = davies_bouldin_score(data, clusters)
        print(f"{method} - Davies-Bouldin Score: {davies_bouldin_avg:.4f}")
    else:
        print("Nie można obliczyć dla jednego klastra")