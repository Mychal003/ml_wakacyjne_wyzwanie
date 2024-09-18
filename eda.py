import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io

# wczytanie danych:
def load_data(file_path):
    with io.open(file_path, encoding='utf-8') as file:
        data = json.load(file)
    df = pd.json_normalize(data)
    return df

def basic_info(df):
    # Wyświetla podstawowe informacje o danych
    print(f'Podstawowe informacje o danych: {df.info()}\n')
    print(f'Podgląd danych: {df.head()}\n')
    print(f'Opis statystyczny: {df.describe()}\n')

def check_missing_data(df):
    # Sprawdza brakujące wartości w zbiorze danych
    missing_values = df.isnull().sum()
    print("\nBrakujące wartości w danych:")
    print(missing_values[missing_values > 0])

def plot_category_distribution(df):
    # Wizualizacja rozkładu kategorii koktajli
    plt.figure(figsize=(10, 6))
    sns.countplot(y="category", data=df, order=df["category"].value_counts().index)
    plt.title("Rozkład kategorii koktajli")
    plt.show()

def plot_glass_distribution(df):
    # Wizualizacja typów szkła
    plt.figure(figsize=(10, 6))
    sns.countplot(y="glass", data=df, order=df["glass"].value_counts().index)
    plt.title("Rozkład typów szkła")
    plt.show()

def alcoholic_distribution(df):
    # Wizualizacja rozkładu z i bez alkoholu
    plt.figure(figsize=(6, 4))
    sns.countplot(x="alcoholic", data=df)
    plt.title("Rozkład koktajli alkoholowych i bezalkoholowych")
    plt.xticks([0, 1], ["Bezalkoholowe", "Alkoholowe"])
    plt.show()
'''
def main():
    file_path = 'C:/Users/pawli/OneDrive/Pulpit/cocktail_dataset.json'# z katalogu projektu po nazwie nie czytało
    df = load_data(file_path)
    basic_info(df)
    check_missing_data(df)
    plot_category_distribution(df)
    plot_glass_distribution(df)
    alcoholic_distribution(df)

if __name__ == "__main__":
    main()
'''