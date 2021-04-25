# TP3-SIA 2021 1er C.

Implementaciones de perceptrón simple y multicapa.
```
Integrantes:
- Eugenia Sol Piñeiro
- Scott Lin
- Nicolás Comerci Wolcanyik
```

### Requerimientos previos
- Python 3


### Instalación

Ejecutar el siguiente comando en la terminal para instalar los módulos de python requeridos
```bash
$> pip install -r requirements.txt
```

### Guía de uso
En el archivo `config.json` del directorio raíz se encuentran los distintos parámetros para configurar la ejecución. Estos son:

- `perceptron_type`: Perceptrón a utilizar (step_simple_perceptron, linear_simple_perceptron, non_linear_simple_perceptron, multilayer_perceptron)

- `learning_rate`: Tasa de aprendizaje (Ej: 0.01)
- `max_iterations`: Cantidad máxima de iteraciones (solo se toma en cuenta cuando se utiliza un perceptrón simple)
- `training_amount`: porcentaje del set de entrenamiento que se utiliza para entrenar a la red (Ej: 0.8)
- `multilayer_perceptron`: parámetros de configuración del perceptrón multicapa
  - `hidden_layers`: lista la cual su longitud representa la cantidad de capas ocultas y los valores en cada posición representa la cantidad de neuronas en dicha capa (Ej: [5, 3] son 2 capas ocultas en las cuales hay 5 neuronas en la primera y 3 en la segunda)
  - `epochs_amount`: cantidad de épocas
  - `batch`: booleano para batch
  - `adaptive_eta`: booleano para ETA adaptativo
  - `momentum`: booleano para momentum
- `training_file_path`: path del txt del set de entrenamiento (Ej: 'files/ej1_training_set.txt')
- `training_file_lines_per_entry`: cantidad de líneas del txt del set de entrenamiento que se toman como una sola entrada (dejarlo siempre en 1 y para el ej3 de los píxeles ponerlo en 7)
- `output_file_path`: path del txt de las salidas esperadas (Ej: 'files/xor_expected_output.txt')

Para ejecutar el programa, correr el siguiente comando en consola:
```bash
$> python3 main.py
```

### POSIBLES CONFIGURACIONES 
#### EJERCICIO 1 
##### AND 
{
    "perceptron_type": "step_simple_perceptron",
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "training_amount": 1,
    "multilayer_perceptron": {
        "hidden_layers": [10,10],
        "epochs_amount": 1000,
        "batch": true,
        "momentum": true,
        "adaptive_eta": {
            "use": false,
            "increase_by": 0.01,
            "decrease_by": 0.03,
            "max_iterations": 4
        }        
    },
    "cross_validation": false,
    "training_file_path": "files/ej1_training_set.txt",
    "training_file_lines_per_entry": 1,
    "output_file_path": "files/and_expected_output.txt"
}
##### XOR 
{
    "perceptron_type": "step_simple_perceptron",
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "training_amount": 1,
    "multilayer_perceptron": {
        "hidden_layers": [10,10],
        "epochs_amount": 1000,
        "batch": true,
        "momentum": true,
        "adaptive_eta": {
            "use": false,
            "increase_by": 0.01,
            "decrease_by": 0.03,
            "max_iterations": 4
        }        
    },
    "cross_validation": false,
    "training_file_path": "files/ej1_training_set.txt",
    "training_file_lines_per_entry": 1,
    "output_file_path": "files/xor_expected_output.txt"
}
-------------
#### EJERCICIO 2 
{
    "perceptron_type": "non_linear_simple_perceptron",
    "learning_rate": 0.06,
    "max_iterations": 1000,
    "training_amount": 1,
    "multilayer_perceptron": {
        "hidden_layers": [10,10],
        "epochs_amount": 10000,
        "batch": false,
        "momentum": false,
        "adaptive_eta": {
            "use": false,
            "increase_by": 0.01,
            "decrease_by": 0.03,
            "max_iterations": 4
        }        
    },
    "cross_validation": false,
    "training_file_path": "files/ej2_training_set.txt",
    "training_file_lines_per_entry": 1,
    "output_file_path": "files/ej2_expected_output.txt"
}
-------------
#### EJERCICIO 3 
##### XOR 
{
    "perceptron_type": "multilayer_perceptron",
    "learning_rate": 0.02,
    "max_iterations": 1000,
    "training_amount": 1,
    "multilayer_perceptron": {
        "hidden_layers": [10,15],
        "epochs_amount": 1000,
        "batch": true,
        "momentum": false,
        "adaptive_eta": {
            "use": true,
            "increase_by": 0.01,
            "decrease_by": 0.03,
            "max_iterations": 4
        }        
    },
    "cross_validation": false,
    "training_file_path": "files/ej1_training_set.txt",
    "training_file_lines_per_entry": 1,
    "output_file_path": "files/xor_expected_output.txt"
}
##### MAPA DE PIXELES
{
    "perceptron_type": "multilayer_perceptron",
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "training_amount": 1,
    "multilayer_perceptron": {
        "hidden_layers": [10,10],
        "epochs_amount": 1000,
        "batch": true,
        "momentum": true,
        "adaptive_eta": {
            "use": false,
            "increase_by": 0.01,
            "decrease_by": 0.03,
            "max_iterations": 4
        }        
    },
    "cross_validation": false,
    "training_file_path": "files/ej3_mapa_de_pixeles.txt",
    "training_file_lines_per_entry": 7,
    "output_file_path": "files/ej3_mapa_output.txt"
}