# 🧠 Laboratorio 3 – Planning with Qwen3-8B

## 📌 Overview

Este proyecto implementa algoritmos de planificación simbólica utilizando **Qwen3-8B** para resolver problemas lógicos de múltiples pasos dentro de una simulación virtual.

El sistema:

- Lee escenarios desde `Task.json`
- Genera secuencias de acciones óptimas
- Calcula automáticamente el nivel de complejidad
- Produce un archivo `submission.json`

---

## ⚙️ Restricciones del laboratorio

✔ Solo se permite **Qwen3-8B**  
✔ No se permite fine-tuning  
✔ Inferencia determinista (`temperature=0.0`)  
✔ Tiempo máximo de ejecución < 2 minutos en Colab  
✔ Las salidas deben ser reproducibles para auditoría  

---

## 📂 Estructura del Proyecto

```
.
├── Examples.json        # Ejemplos few-shot con soluciones óptimas
├── Task.json            # Dataset de evaluación (solo escenarios)
├── planning.ipynb          # Script principal
├── submission.json      # Archivo generado para enviar
└── README.md
```

---

## 🏗️ Arquitectura del Enfoque

Se utiliza una arquitectura **Few-Shot Prompting** con:

- Separación automática de dominios:
  - `set of blocks`
  - `set of objects`
- 2–3 ejemplos relevantes por dominio
- Generación determinista
- Validación robusta de JSON

El modelo devuelve únicamente:

```json
{
  "complexity_level": 4,
  "target_action_sequence": [
    "(engage_payload a)",
    "(unmount_node a b)",
    "(mount_node a c)",
    "(release_payload a)"
  ]
}
```


## 🚀 Cómo Ejecutarlo en Google Colab

### 1️⃣ Activar GPU

Runtime → Change runtime type → GPU

---

### 2️⃣ Instalar dependencias

```python
!pip install transformers accelerate
```

---

### 3️⃣ Cargar modelo

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
```

---

### 4️⃣ Ejecutar generación

```python
main(model, tokenizer)
```

Se generará:

```
submission.json
```

---

## 👤 Grupo - OptimusPrime:

* César Eduardo Pajuelo Reyes
* Gonzalo Alonso Rodriguez Gutierrez
