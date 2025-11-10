# Air-Quality-Forecast

Krótki przewodnik, jak przygotować środowisko i wystartować z pracą nad projektem.

Wymagania wstępne:
- Python: 3.10–3.12 (sprawdź wersję: python --version)
- Git: do pracy z repozytorium
- (Opcjonalnie) Make, PyCharm / VS Code
- (Opcjonalnie) Git LFS jeśli będziesz przechowywać duże pliki (modele/dane > 100 MB)

1) Pobranie danych
- Pobierz przygotowane zbiory danych (cache) spod tego linku: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data 
- Rozpakuj archiwum do struktury katalogów projektu:   
```kotlin
.
└─ data/
   └─ cache/
      ├─ mlp_dataset.npz
      └─ lstm_dataset.npz

```
- Upewnij się, że ścieżki w config.yaml wskazują na właściwe pliki:

2) Klon repozytorium
```
git clone https://github.com/ORG_OR_USER/REPO.git
cd REPO
```

3) Utwórz i aktywuj środowisko wirtualne:
```
python -m venv .venv
venv\Scripts\Activate.ps1
```
4) Instalacja zależności
```
pip install --upgrade pip
pip install -r requirements.txt
```

5) Plik konfiguracyjny config.yaml   
Projekt używa jednego, czytelnego pliku YAML do ustawień ścieżek i treningu.
Utwórz plik config.yaml w katalogu głównym repo na podstawie poniższego wzoru.

```yaml
# config.yaml — główna konfiguracja projektu

project:
  seed: 42
  output_dir: "outputs"
  models_dir: "models"
  cache_dir: "data/cache"

data:
  # Ścieżki do przygotowanych zestawów (np. pliki .npz)
  mlp_dataset: "data/cache/mlp_dataset.npz"
  lstm_dataset: "data/cache/lstm_dataset.npz"
  # Podstawowe info o danych (pomocne do walidacji)
  frequency: "1H"          # np. 1H, 10min
  target: "pm25"           # nazwa zmiennej docelowej
  features:                 # opcjonalnie lista kluczowych cech
    - temp
    - humidity
    - wind_speed

training:
  # Ustawienia wspólne
  epochs: 100
  batch_size: 256
  early_stopping_patience: 10

  # MLP (wektorowe cechy)
  mlp:
    use: true
    dataset: "@data.mlp_dataset"  # odwołanie do klucza z sekcji data
    optimizer: "adam"
    loss: "mse"
    metrics: ["mae", "rmse", "r2"]

  # LSTM (dane sekwencyjne)
  lstm:
    use: true
    dataset: "@data.lstm_dataset"
    timesteps: 48
    optimizer: "adam"
    loss: "mse"
    metrics: ["mae", "rmse", "r2"]
```

6) Szybki start (przykład)    
Zakładając, że masz przygotowane pliki .npz i train.py obsługujący --model oraz --npz-path:
```bash
# MLP
python train.py --model mlp --data npz --npz-path "$(yq '.data.mlp_dataset' config.yaml)" --epochs "$(yq '.training.epochs' config.yaml)"

# LSTM
python train.py --model lstm --data npz --npz-path "$(yq '.data.lstm_dataset' config.yaml)" --epochs "$(yq '.training.epochs' config.yaml)"
```

7) Struktura katalogów (proponowana)    
```arduino
.
├─ data/
│  └─ UCI Beijing
│  └─ cache/
│     ├─ mlp_dataset.npz
│     └─ lstm_dataset.npz
├─ models/
├─ outputs/
├─ train.py
├─ config.yaml
├─ requirements.txt
└─ README.md
```

9) Typowe problemy    
- Błąd przy instalacji TensorFlow/PyTorch: sprawdź zgodność wersji z Pythonem i (jeśli używasz GPU) z CUDA/CuDNN.
- Brak yq: zainstaluj (brew install yq, apt-get install yq) lub podaj wartości z config.yaml ręcznie.
- pip instaluje długo: upewnij się, że masz aktualnego pip i używasz lokalnego ~/.cache/pip.