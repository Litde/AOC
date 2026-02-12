import os
import matplotlib.pyplot as plt

def plot_category_histogram(root_path):
    """
    Rekurencyjnie przeszukuje podfoldery kategorii
    i zlicza wszystkie pliki .jpg/.png na dowolnym poziomie.
    """

    category_counts = {}

    # pierwszy poziom = kategorie
    for category in os.listdir(root_path):
        category_path = os.path.join(root_path, category)

        if os.path.isdir(category_path):
            image_count = 0

            # rekurencyjne przejście po całym drzewie folderu
            for dirpath, dirnames, filenames in os.walk(category_path):
                for file in filenames:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_count += 1

            category_counts[category] = image_count

    # dane do wykresu
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # wykres
    plt.figure()
    plt.bar(categories, counts)
    plt.xlabel("Kategorie")
    plt.ylabel("Liczba obrazów")
    plt.title("Histogram liczby obrazów w kategoriach")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return category_counts

if __name__ == "__main__":
    plot_category_histogram("input")
