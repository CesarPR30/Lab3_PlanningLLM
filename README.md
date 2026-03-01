# 🧠 Laboratorio 3 – Planning con Qwen3-8B

Este repositorio contiene la solución para el **Laboratorio 3: Planning**, donde se debe diseñar un agente capaz de resolver problemas lógicos de múltiples pasos dentro de una simulación virtual usando **Qwen3-8B** con inferencia determinista.

El agente recibe escenarios descritos en lenguaje natural y genera una **secuencia de acciones** para alcanzar el objetivo.

---
## Colab

Link: 
https://colab.research.google.com/drive/1MCz9s9pYaYHUccPanRlkV3Wd3WWHIMHF?usp=sharing




---

## 🎯 Objetivo

Diseñar un agente que:

- Use exclusivamente **Qwen3-8B**
- Use inferencia determinista (**temperature=0.0**, `do_sample=False`)
- Respete el límite de **2 minutos por task en Colab**
- Procese `Task.json` y genere un `submission.json`
- Calcule y devuelva:
  - `complexity_level`
  - `target_action_sequence`

---

## 📁 Estructura del repositorio

```
submit.py        -> Genera submission.json
evaluator.py     -> Métrica usada en la evaluación
student_agent.py -> Implementación del agente (archivo principal evaluado)
llm_engine.py    -> Carga de Qwen3-8B y wrapper de inferencia
dev_test.py      -> Script para probar el agente y ver score
Examples.json    -> Dataset de desarrollo con soluciones óptimas
Task.json        -> Dataset de evaluación (solo escenarios)
colab.ipynb      -> Notebook con todo integrado para correr en Colab
README.md        -> Este archivo
```

⚠️ Importante:

El archivo que se revisa en la auditoría es:

```
student_agent.py
```

---

## 🧠 Arquitectura del agente

La solución usa:

- Prompt Engineering con reglas estrictas
- Few-shot prompting usando Examples.json
- Retrieval por similitud (Jaccard)
- Separación por dominio:
  - Objects domain
  - Blocks domain
- Inferencia determinista con Qwen3-8B
- Generación de planes mínimos

El agente:

1. Detecta el dominio (blocks / objects)
2. Extrae el último STATEMENT
3. Busca ejemplos similares
4. Construye prompt con reglas
5. Llama a Qwen3-8B
6. Devuelve lista de acciones

---

## ✅ Configuración obligatoria del modelo

El laboratorio exige:

```
temperature = 0.0
do_sample = False
top_p = 1.0
```

Ejemplo:

```python
resp = qwen(
    prompt=prompt,
    system=system,
    temperature=0.0,
    do_sample=False,
    top_p=1.0,
    max_new_tokens=256,
    enable_thinking=False,
    stream=False
)
```

Esto asegura:

- reproducibilidad
- auditoría correcta
- leaderboard válido

---

## 🚀 Cómo ejecutar

### 1) Test en desarrollo

```
python dev_test.py
```

Esto:

- carga Examples.json
- ejecuta el agente
- calcula score
- muestra tiempo por task

---

### 2) Generar submission

```
python submit.py
```

Esto:

- lee Task.json
- ejecuta todos los tasks
- crea submission.json

Formato esperado:

```
[
  {
    "assembly_task_id": "...",
    "complexity_level": 4,
    "target_action_sequence": [
      "(attack a)",
      "(overcome a b)"
    ]
  }
]
```

---

### 3) Ejecutar en Colab

Abrir:

```
colab.ipynb
```

Este notebook contiene:

- instalación
- carga del modelo
- ejecución de tests
- generación de submission.json

---

## ⏱ Restricción de tiempo

Máximo permitido:

```
2 minutos por task
```

Para cumplirlo:

- pocos shots
- prompts compactos
- max_new_tokens limitado
- temperature = 0

---

## 🔍 Auditoría

El profesor verificará:

- que se use Qwen3-8B
- que temperature = 0
- que las salidas sean deterministas
- que student_agent.py produzca lo mismo

Por eso el código usa:

```
do_sample=False
temperature=0.0
top_p=1.0
```

---

## 📊 Estrategias usadas

- Few-shot retrieval
- Prompt rules estrictas
- Domain-specific prompting
- Deterministic decoding
- Minimal plan bias
- Goal-focused constraints

Esto mejora el score sin romper las reglas.

---

## 💻 Colab

Link:  
https://colab.research.google.com/drive/1MCz9s9pYaYHUccPanRlkV3Wd3WWHIMHF?usp=sharing

---


## 👤 Grupo - OptimusPrime

- César Eduardo Pajuelo Reyes
- Gonzalo Alonso Rodriguez Gutierrez
