# VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Proyecto 2 del Colegio Universitario de Cartago (CUC).

VectorIA es un sistema que combina tres modelos de redes neuronales para apoyar la vigilancia del dengue en Costa Rica:

| Módulo | Modelo | Qué hace |
|---|---|---|
| 1 | CNN | Identifica la especie del mosquito en una foto (*Aedes aegypti*, *Aedes albopictus* o *Culex*) |
| 2 | RNN / LSTM | Pronostica los casos de dengue de las próximas 4 semanas usando datos del clima |
| 3 | ANN | Recomienda la acción preventiva según el pronóstico, el clima y la especie detectada |

Una app en Streamlit une los tres modelos y muestra un mapa de calor de casos por cantón.

> **Aviso:** esta es una herramienta de apoyo a la vigilancia. No diagnostica dengue en personas.

## Integrantes

- Mariana Méndez Pérez
- Luis Diego Montero Vargas
- Nadin Rojas López
- Claret Rodríguez Jiménez