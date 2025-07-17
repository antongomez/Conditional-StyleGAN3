# Multispectral Image Processing Utilities

Una biblioteca completa para procesar imágenes multiespectrales y dividir conjuntos de datos en train/validation/test para aplicaciones de machine learning.

## Características

- **Lectura de datos multiespectrales**: Soporte para archivos RAW, PGM, y centros de segmentación
- **División automática de datos**: Splitting inteligente en train/validation/test con balanceo de clases
- **Extracción de patches**: Extracción eficiente de patches con validación de límites
- **Guardado flexible**: Soporte para múltiples formatos (JSON, Pickle, NPZ)
- **Validación de integridad**: Verificación automática de la integridad del dataset
- **Reportes detallados**: Generación de reportes completos con estadísticas

## Estructura del Proyecto

```
multispectral_utils/
├── __init__.py              # Importaciones principales
├── data_readers.py          # Lectores de datos multiespectrales
├── data_splitter.py         # Utilidades para dividir datasets
├── patch_extractor.py       # Extracción y guardado de patches
├── multispectral_utils.py   # Funciones de alto nivel
├── example_usage.py         # Ejemplos de uso
└── README.md               # Esta documentación
```

## Instalación

```bash
# Clona el repositorio
git clone <tu-repositorio>

# Instala las dependencias
pip install numpy torch scikit-learn PIL tqdm
```

## Uso Básico

### Procesamiento completo de un dataset

```python
from multispectral_utils import process_multispectral_dataset

# Procesar dataset completo
results = process_multispectral_dataset(
    input_dir="data/oitaven/",
    filename="oitaven",
    output_dir="data/oitaven/patches/",
    train_size=0.15,
    val_size=0.05,
    patch_size=32,
    rgb=False,
    seed=42
)
```

### Cargar un dataset procesado

```python
from multispectral_utils import load_processed_dataset

# Cargar dataset previamente procesado
dataset = load_processed_dataset("data/oitaven/patches/", "json")
```

### Validar integridad del dataset

```python
from multispectral_utils import validate_dataset_integrity

# Validar integridad
validation = validate_dataset_integrity("data/oitaven/patches/", "json")
print(f"Dataset válido: {validation['valid']}")
```

## Uso por Módulos

### 1. Lectura de Datos (`data_readers.py`)

```python
from multispectral_utils import load_multispectral_dataset

# Cargar dataset completo
dataset = load_multispectral_dataset("data/oitaven/", "oitaven")
print(f"Dimensiones: {dataset['dimensions']}")
```

### 2. División de Datos (`data_splitter.py`)

```python
from multispectral_utils import split_dataset, save_split_info

# Dividir dataset
train, val, test, label_map = split_dataset(
    truth=dataset['truth'],
    centers=dataset['centers'],
    image_height=dataset['dimensions']['height'],
    image_width=dataset['dimensions']['width'],
    patch_width=32,
    patch_height=32,
    train_size=0.15,
    val_size=0.05,
    seed=42
)

# Guardar información de la división
save_split_info(train, val, test, label_map, "split_info.json")
```

### 3. Extracción de Patches (`patch_extractor.py`)

```python
from multispectral_utils import batch_extract_patches

# Extraer patches para todos los splits
json_paths = batch_extract_patches(
    data=dataset['data'],
    truth=dataset['truth'],
    train_centers=train,
    val_centers=val,
    test_centers=test,
    image_height=dataset['dimensions']['height'],
    image_width=dataset['dimensions']['width'],
    patch_size=32,
    base_output_dir="output/",
    rgb=False
)
```

## Formato de Archivos de División

### Recomendaciones por Formato

1. **JSON** (Recomendado para la mayoría de casos):

   - ✅ Legible por humanos
   - ✅ Compatible con diferentes lenguajes
   - ✅ Fácil de versionar
   - ❌ Archivos más grandes

2. **Pickle** (Para datasets muy grandes):

   - ✅ Más compacto
   - ✅ Preserva tipos de datos Python
   - ❌ Solo compatible con Python
   - ❌ No legible por humanos

3. **NPZ** (Para arrays grandes):
   - ✅ Muy eficiente para arrays numpy
   - ✅ Compresión automática
   - ❌ Menos flexible para metadatos

### Contenido del Archivo de División

Cada archivo de división contiene:

```python
{
    "train_indices": [1, 5, 10, ...],           # Índices de entrenamiento
    "validation_indices": [2, 7, 15, ...],     # Índices de validación
    "test_indices": [3, 8, 20, ...],           # Índices de test
    "label_map": {1: 1, 2: 2, ...},            # Mapeo de clases
    "metadata": {                               # Metadatos
        "dataset_name": "oitaven",
        "patch_size": 32,
        "train_size": 0.15,
        "processing_date": "2024-01-01T10:00:00"
    },
    "split_stats": {                            # Estadísticas
        "train_samples": 1500,
        "validation_samples": 500,
        "test_samples": 8000,
        "total_samples": 10000,
        "num_classes": 10
    }
}
```

## Ejemplo de Línea de Comandos

```bash
# Procesar dataset completo
python example_usage.py --mode process \
                       --input-dir data/oitaven/ \
                       --filename oitaven \
                       --output-dir data/oitaven/patches/ \
                       --train-size 0.15 \
                       --validation-size 0.05 \
                       --patch-size 32 \
                       --seed 42

# Generar reporte
python example_usage.py --mode report --output-dir data/oitaven/patches/

# Validar integridad
python example_usage.py --mode validate --output-dir data/oitaven/patches/
```

## Estructura de Salida

```
output_dir/
├── split_info.json                    # Información de división
├── processing_summary.json            # Resumen del procesamiento
├── dataset_report.txt                 # Reporte detallado
├── train/
│   ├── dataset.json                   # Etiquetas de entrenamiento
│   └── 00001/                         # Carpeta por clase
│       ├── img00000001.npy           # Patches guardados
│       └── img00000002.npy
├── validation/
│   ├── dataset.json
│   └── 00001/
│       └── img00000001.npy
└── test/
    ├── dataset.json
    └── 00001/
        └── img00000001.npy
```

## Consideraciones Importantes

### Para División de Datos

- Los centros identifican unívocamente cada patch
- Se garantiza que no hay solapamiento entre splits
- El balanceo de clases se mantiene en train/validation
- Todos los samples válidos van a test (incluyendo train y validation)

### Para Extracción de Patches

- Se validan los límites de imagen automáticamente
- Solo se procesan patches con etiquetas válidas (> 0)
- Normalización automática de datos
- Soporte para modo RGB (canales 2,1,0)

### Para Reproducibilidad

- Usar siempre el mismo seed para resultados consistentes
- Guardar metadatos completos en archivos de división
- Validar integridad después del procesamiento
