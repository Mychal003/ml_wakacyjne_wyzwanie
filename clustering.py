from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
# Rozpakowanie składników do listy nazw
def extract_ingredient_names(ingredients):
    """
    Rozpakowuje listę słowników ze składnikami i zwraca tylko nazwy składników.
    """
    return [ingredient['name'] for ingredient in ingredients]

def preprocess_data(df):
    # - Krok 1: Rozpakowanie i przekształcenie składników ---
    df['ingredient_names'] = df['ingredients'].apply(extract_ingredient_names)

    # Przekształcenie list składników w macierz 0/1
    mlb = MultiLabelBinarizer()
    ingredient_matrix = mlb.fit_transform(df['ingredient_names'])

    # Przekształcenie na DataFrame
    ingredient_df = pd.DataFrame(ingredient_matrix, columns=mlb.classes_, index=df.index)

    # - Krok 2: OneHotEncoding dla kategorii ---
    categorical_columns = ["category", "glass"]

    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(df[categorical_columns])

    # Stworzenie DataFrame z zakodowanymi zmiennymi kategorycznymi
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, 
                                          columns=encoder.get_feature_names_out(categorical_columns),
                                          index=df.index)

    # - Krok 3: Scalanie przekształconych danych ---
    df_full = pd.concat([df, encoded_categorical_df, ingredient_df], axis=1)

    # Usunięcie niepotrzebnych kolumn tekstowych oraz innych nienumerycznych (np. tags)
    df_full.drop(columns=['name', 'instructions', 'imageUrl', 'createdAt', 'updatedAt', 
                          'ingredients', 'ingredient_names', 'category', 'glass', 'tags'], inplace=True)

    # Sprawdzenie typów danych
    print("Typy danych w df_full:")
    print(df_full.dtypes)

    # Podgląd danych
    print(df_full.head())

    # --- Krok 4: Skalowanie danych ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_full)

    return scaled_data

def kmeans_clustering(data, n_clusters=5):
    """
    Przeprowadza klasteryzację K-means.
    - n_clusters: liczba klastrów (domyślnie 5).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    print(f'K-means: Zidentyfikowane klastry: {n_clusters}')
    return clusters

def dbscan_clustering(data, eps=65.0, min_samples=3):
    """
    Przeprowadza klasteryzację DBSCAN.
    - eps: maksymalna odległość między dwoma punktami, aby mogły być uznane za sąsiednie.
    - min_samples: minimalna liczba punktów wymaganych do utworzenia klastra.
    Zwraca etykiety klastrów.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # -1 to szum
    print(f'DBSCAN: Zidentyfikowane klastry: {n_clusters}')
    return clusters
