# Testy eksperymentalne dla FVHD

W katalogu `tests/` znajdują się skrypty służące do oceny działania algorytmu **FVHD** oraz jego rozszerzonej wersji w różnych wariantach.

---

## 'test_distilled.py' - wersja z destylacją danych

Ten test:

- Wczytuje zestaw **1000 prototypów** z pliku `.npz`, które powstały ze skryptu pochądzącego z literatury i są to dane zdestylowane z mnist np. 
- Trenuje model `FVHDWithTransform` wyłącznie na tych danych (`fit(...)`) syntetycznych
- Wczytuje pełny zbiór **MNIST** (60 000 przykładów)
- Rzutuje pełne dane do przestrzeni embeddingów prototypów przy użyciu metody `.project(...)`
- Wylicza **silhouette score** oraz zapisuje wizualizacje

### Jak to działa - fit + transform

Dla każdego nowego punktu:

1. Obliczane są odległości do wszystkich prototypów
2. Wybieranych jest `top_k` najbliższych
3. Embeddingi tych prototypów są **uśredniane (ważone odwrotnością odległości)**
4. Wynikiem jest rzut nowego punktu w przestrzeni 2D

---

## 'tests_impovement.py' - testy wariantów algorytmu i ulepszeń

Ten skrypt:

- Wczytuje cały zbiór **MNIST**
- Tworzy graf sąsiedztwa (k-NN)
- Trenuje model `FVHD` metodą `.fit_transform(...)`
- Testuje wiele wariantów konfiguracyjnych
- Wyniki (czas działania, silhouette score) są zapisywane do CSV

### Testowane opcje

| Parametr                  | Opis                                                                 |
|---------------------------|----------------------------------------------------------------------|
| 'c'                       | Stała regulująca siłę odpychania między punktami                     |
| 'eta_schedule'            | Zmienny learning rate (np. 'decay', 'cosine')                        |
| 'mutual_neighbors_epochs' | Liczba epok, przez które używani są tylko wzajemni sąsiedzi          |
| 'gaussian_weights'        | Użycie wag Gaussa dla sił przyciągających                            |
| 'autoadapt`, `boost_eta'  | Dynamiczne dostosowanie siły sił (lub początkowego 'eta')            |
| 'velocity_limit'          | Ograniczenie maksymalnej prędkości punktów w przestrzeni             |
| 'supervised'              | Użycie etykiet klas do wzmacniania struktury embeddingu              |

---

## 'test_supervised.py' - embeddingi z nadzorem (`supervised=True`)

Ten skrypt testuje działanie modelu `FVHD` lub `FVHDWithTransform` w trybie **nadzorowanym**.

### Jak t działa

- W treningu uwzględniane są etykiety klas (`labels`)
- Siły przyciągające między punktami tej samej klasy są **wzmacniane**
- Siły odpychające między punktami z różnych klas mogą być **zwiększane**
- W funkcji straty mogą wystąpić dodatkowe składniki:
  - `lambda1`, `lambda2` – regulujące wpływ części nadzorowanej

Dzięki temu embeddingi są lepiej dopasowane klasowo – punkty z tej samej etykiety grupują się bliżej, co poprawia np. silhouette score lub klasyfikację.

---

## Wyniki

- Wszystkie testy zapisują wyniki do: `results/summary.csv`
- Wizualizacje są zapisywane jako: `results/<NazwaTestu>.png`

---
