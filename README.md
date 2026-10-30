# NuraFace

Aplikacja desktopowa do rejestracji użytkowników i weryfikacji tożsamości
z wykorzystaniem obrazu z kamery, rozpoznawania twarzy oraz dodatkowych
metod uwierzytelniania.

> Projekt badawczy. Nie należy używać go jako jedynego zabezpieczenia
> systemu ani do podejmowania decyzji o istotnych skutkach dla ludzi.

## Wymagania

- Windows
- Python 3.12
- Kamera internetowa

## Instalacja i uruchomienie

W PowerShell, w katalogu projektu:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Dane i modele

Wyniki eksperymentów H1 i H2 znajdują się w katalogach `wyniki_h1/` i
`wyniki_h2/`. W szczegółowym CSV H2 ścieżki do plików są zapisane
względem katalogu projektu. Katalog `database/` w repozytorium zawiera
jedynie plik `.gitkeep` i nie zawiera danych.

Zbiory zdjęć i pliki danych nie są przechowywane w tym repozytorium.
W celu przeprowadzenia eksperymentów wymagane pliki należy pozyskać
z oficjalnych źródeł, z uwzględnieniem warunków ich wykorzystania, a
następnie umieścić lokalnie w katalogach `database/` i `test_database/`.
Zawartość obu katalogów jest wykluczona z repozytorium za pomocą
`.gitignore`, dlatego po sklonowaniu repozytorium wymagane pliki należy
skopiować lub przygotować lokalnie. Pliki `ff_train.csv` i `ff_test.csv`
powinny znajdować się w obu katalogach. W obu katalogach wymagana jest
również struktura `verification/`; dodatkowo w `test_database/` wymagany
jest katalog `profile/`. Szczegóły przedstawiono poniżej. Przed dalszym
udostępnianiem danych należy zweryfikować licencje oraz warunki
redystrybucji.

Wagi modeli nie są dołączone do repozytorium. Wymagane pliki można pozyskać
z poniższych źródeł i zapisać lokalnie pod nazwami oczekiwanymi przez
aplikację:

| Plik | Źródło | SHA-256 |
|---|---|---|
| `weights/yolo_face.pt` | [YOLOv8n-Face — Lindevs](https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt) | `b038ca653b503453a94f6e12d76feca6840b2a97d7a1322b4498c5e922f29832` |
| `weights/arcface.onnx` | [ArcFace ONNX — Hugging Face](https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true) | `ffe014a45c9488506719d37fd578ece6661bb385535b36e8039975fa5d4683db` |

Pobranie plików w PowerShell z katalogu projektu:

```powershell
New-Item -ItemType Directory -Force weights
Invoke-WebRequest "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt" -OutFile "weights/yolo_face.pt"
Invoke-WebRequest "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -OutFile "weights/arcface.onnx"
```

Weryfikacja sum kontrolnych:

```powershell
Get-FileHash weights/yolo_face.pt -Algorithm SHA256
Get-FileHash weights/arcface.onnx -Algorithm SHA256
```

Przed użyciem lub dalszym rozpowszechnianiem modeli należy zweryfikować
licencje i warunki ich dostawców. Zdjęcia twarzy, dane uwierzytelniające
oraz logi nie powinny być publikowane w repozytorium.

### Dane testowe

Zbiory danych są przechowywane poza repozytorium ze względu na ich rozmiar
oraz zawartość zdjęć twarzy. Pliki należy pozyskać z oficjalnych źródeł
zgodnie z warunkami ich wykorzystania i umieścić lokalnie w katalogu
`test_database/`. Pobierane dane nie są częścią publicznego repozytorium.

Aplikacja i skrypty analityczne oczekują następującej struktury katalogów:

```text
database/
├── ff_train.csv
├── ff_test.csv
└── verification/
    └── <id>/
        └── vanilla/
            └── pos_<n>/
                └── angle_<n>/
                    └── IMG_<n>.JPG

test_database/
├── ff_train.csv
├── ff_test.csv
├── profile/
│   └── <id>/
│       ├── <id>.csv
│       └── IMG_<n>.JPG
└── verification/
    └── <id>/
        └── vanilla/
            └── pos_<n>/
                └── angle_<n>/
                    └── IMG_<n>.JPG
```

W eksperymentach wykorzystano zbiór FairFace jako źródło próbek impostorów
oraz Facial Recognition Dataset FULL jako źródło zdjęć testowych.
Wykorzystanie obu zbiorów powinno odbywać się zgodnie z ich licencjami
i warunkami dostępu.

## Struktura projektu

- `app/` — logika aplikacji i interfejs użytkownika
- `extraction/` — ekstrakcja cech
- `modeling/` — trenowanie i ewaluacja modeli
- `verification/` — weryfikacja
- `resources/` — konfiguracja i zasoby interfejsu
- `main.py` — punkt startowy aplikacji

## Budowanie wersji Windows

Skrypt `build.bat` buduje aplikację za pomocą PyInstaller. Wymaga
przygotowanych lokalnie zależności, danych i wag modeli. Wygenerowane
pliki w `build/` i `dist/` nie są częścią repozytorium.

## Licencje i źródła

Przed dalszym rozpowszechnianiem kodu, modeli lub danych należy zweryfikować
ich licencje oraz warunki wykorzystania. Licencje modeli i zbiorów danych
są niezależne od licencji kodu aplikacji.
