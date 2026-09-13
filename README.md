# Agente DAE - Continuidad y Becas

Agente conversacional basado en RAG (Retrieval-Augmented Generation) que responde
consultas de estudiantes sobre requisitos de becas, continuidad de estudios y
fechas del calendario académico, usando únicamente información reglamentaria
como fuente de verdad.

## ¿Qué problema resuelve?

La Dirección de Asuntos Estudiantiles (DAE) recibe muchas consultas repetitivas
sobre becas y continuidad, sobre todo en cierre/inicio de semestre. Este agente
automatiza esas respuestas basándose en normativa real, evitando que la IA
invente promedios, fechas o requisitos.

## ¿Cómo funciona? (arquitectura RAG)

1. **Fuentes de información**: se simulan 3 documentos institucionales
   (`simular_ingesta_fuentes_institucionales`):
   - Reglamento interno de becas (dato no estructurado, interno).
   - Decreto del MINEDUC sobre gratuidad (dato no estructurado, externo).
   - Calendario académico con fechas clave (dato estructurado).

2. **Fragmentación (chunking)**: cada documento se divide en fragmentos de
   hasta 1200 caracteres con `RecursiveCharacterTextSplitter`, para que la
   búsqueda por similitud sea más precisa.

3. **Embeddings**: cada fragmento se convierte en un vector numérico.
   El script usa **Google Gemini** (`models/embedding-001`) si detecta
   `GOOGLE_API_KEY`; si no, usa OpenAI (`text-embedding-3-small`) como
   respaldo.

4. **Base vectorial**: los vectores se guardan en **ChromaDB** (en memoria),
   junto a sus metadatos (`fuente`, `tipo`, `origen`) para trazabilidad.

5. **Recuperación**: ante cada pregunta, se buscan los **2 fragmentos** más
   parecidos semánticamente en la base vectorial.

6. **Generación**: los fragmentos recuperados se inyectan como contexto en el
   `SYSTEM_PROMPT`, junto a la pregunta del estudiante, y se envían al modelo
   de lenguaje (**Mistral**, vía API compatible con OpenAI) para generar la
   respuesta final.

## Reglas del agente (System Prompt)

- Responde solo con la información del contexto recuperado.
- Si el contexto no tiene la respuesta, deriva a la oficina DAE presencial.
- Nunca inventa promedios, porcentajes ni fechas.
- Siempre recomienda revisar el correo institucional.
- Cita explícitamente la fuente de cada respuesta.

## Estructura del código

| Sección | Función/Clase | Descripción |
|---|---|---|
| Embeddings | `embeddings` | Selección automática Gemini / OpenAI |
| Ingesta | `simular_ingesta_fuentes_institucionales()` | Documentos de ejemplo con metadatos |
| Vectorización | `inicializar_vectorstore()` | Chunking + Chroma |
| Prompt | `SYSTEM_PROMPT`, `prompt_template` | Reglas del agente |
| Orquestación | `AgenteDAE` | Retrieval + generación + trazabilidad |
| Pruebas | bloque `if __name__ == "__main__"` | 3 casos de prueba, incluyendo uno fuera de contexto |

## Requisitos

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters \
    langchain-openai langchain-google-genai chromadb python-dotenv google-generativeai
```

## Variables de entorno

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `LLM_API_KEY` | Key de Mistral (obligatoria) | https://console.mistral.ai/api-keys |
| `LLM_BASE_URL` | URL de la API (por defecto `https://api.mistral.ai/v1`) | — |
| `GOOGLE_API_KEY` | Key de Google AI Studio (opcional, para embeddings gratis) | https://aistudio.google.com/apikey |

## Ejecución

```bash
python agente_dae_rag.py
```

El script corre 3 preguntas de prueba y muestra, para cada una: la respuesta
del agente y los documentos usados como evidencia (fuente, tipo y origen).

## Casos de prueba incluidos

1. Pregunta con respuesta en el reglamento (promedio para mantener la beca).
2. Pregunta con respuesta en el calendario (fecha de matrícula).
3. Pregunta fuera de contexto (saldo JUNAEB) — valida que el agente derive a
   la oficina DAE en vez de inventar una respuesta.
