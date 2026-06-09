# Company Profiler

Pipeline de investigación de empresas en tres fases: **Discovery → Research → Write**.  
Dado un conjunto de requirements, encuentra empresas que los cumplen, investiga su tecnología en profundidad y genera perfiles estructurados pregunta por pregunta.

---

## Arquitectura

```
python main.py discover      →   candidates_<ts>.json  +  discovery_results_<ts>.txt
        ↓ (editás el JSON / el TXT)
python main.py import list.csv   (alternativa: importás tu propia lista)
        ↓
python main.py research      →   research_<ts>.json
        ↓
python main.py write         →   <company>_profile_<ts>.json  (uno por empresa)
```

Cada comando es independiente: podés correr solo `research` o solo `write` si ya tenés los archivos previos en `output/`.

---

## Setup

### 1. Dependencias

```bash
pip install -r requirements.txt
```

### 2. Variables de entorno

Creá un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=sk-...
EXA_API_KEY=...          # para Discovery y Research (recomendado)
PERPLEXITY_API_KEY=pplx-...  # para Writer y Research (opcional pero mejora mucho los resultados)
```

### 3. Archivos de input

| Archivo | Propósito |
|---|---|
| `requirements_example.csv` | Criterios que usa Discovery para buscar empresas |
| `questions_example.csv` | Preguntas que Research extrae y Writer responde |
| `companies_list.csv` | Tu propia lista de empresas (para el comando `import`) |

---

## Comandos

### `discover` — Busca empresas automáticamente

```bash
python main.py discover
```

Lee `requirements_example.csv` y busca empresas que los cumplan usando Exa (con fallback a DuckDuckGo). Genera 15 queries especializadas (LinkedIn, Crunchbase, G2, listas de la industria, etc.) y devuelve hasta 50 candidatos rankeados por score.

**Output:**
- `output/candidates_<ts>.json` — lista editable de empresas con scores y evidence
- `output/discovery_results_<ts>.txt` — versión legible con quotes y fuentes por requirement

**Flujo recomendado:**
1. Revisar `discovery_results_<ts>.txt`
2. Editar `candidates_<ts>.json` para quedarte solo con las empresas que querés investigar
3. Correr `python main.py research`

---

### `import` — Importa tu propia lista de empresas

```bash
python main.py import companies_list.csv
```

Convierte un CSV propio al formato interno y lo guarda como `candidates_<ts>.json`.

**Formato del CSV:**

```csv
name,url,technology_name,technology_url
"Kyhe Technology",https://kyhe.com,DH-S Titanium Powder,https://kyhe.com/product/dh-s-powder/
"Otra Empresa",https://otra.com,,
```

- `technology_name` y `technology_url` son opcionales pero mejoran mucho el research si los tenés
- La columna `technology_url` es la URL específica del producto/tecnología dentro del sitio de la empresa

---

### `research` — Investiga cada empresa

```bash
python main.py research
```

Toma el `candidates_*.json` más reciente de `output/` e investiga cada empresa. Usa tres fuentes para construir la lista de links a crawlear:

1. **Exa site search** — busca dentro del dominio las páginas más relevantes para la tecnología
2. **Spider shallow** — extrae links del homepage (FAQ, About, Contact, etc.)
3. **Perplexity search** — trae fuentes externas con contenido ya incluido (no se crawlean)

**Pausa humana:** antes de crawlear, guarda `output/links_<company>.txt` y abre el archivo automáticamente en Notepad para que puedas revisar, reordenar o eliminar páginas. Presioná Enter para continuar.

Luego crawlea las páginas aprobadas, extrae PDFs, y para cada documento ejecuta un único GPT call que extrae quotes relevantes para todas las preguntas a la vez (extracción estructurada). Finalmente indexa todo en FAISS.

**Output:**
- `output/research_<ts>.json` — metadata del research por empresa
- `data/company_indexes/<company>.faiss` — índice vectorial
- `data/<company>_extracted.json` — quotes estructuradas por pregunta

---

### `write` — Genera perfiles

```bash
python main.py write
```

Toma el `research_*.json` más reciente y genera un perfil por empresa respondiendo cada pregunta. Para cada pregunta combina tres fuentes:

1. **Direct quotes** extraídas durante el research (mayor confianza)
2. **FAISS chunks** — fragmentos semánticamente cercanos del texto crawleado
3. **Perplexity** — respuesta sintetizada de la web (en batches de 5 preguntas para reducir costo)

Si la respuesta inicial es `"No evidence..."`, hace un **fallback a deep-research** de Perplexity para esa pregunta específica.

El tono y formato se calibra con **Style Examples** (CSV en `data/style_examples/`).

**Output:**
- `output/<company>_profile_<ts>.json` — perfil completo con respuestas y fuentes

---

## Archivos de configuración

### `requirements_example.csv`

```csv
type,requirement
must_have,Uses vacuum packaging technology
must_have,Sells packaging machinery to food manufacturers
desirable,Has automatic sealing systems
```

- `type`: `must_have` o `desirable`
- `requirement`: descripción en lenguaje natural

### `questions_example.csv`

```csv
question,description
Company Overview,"Brief description: what the company does, size, markets served"
Technology Type,"What type of packaging technology do they use?"
```

- `question`: nombre corto (se usa como ID)
- `description`: instrucción más larga que guía la extracción y respuesta

### Style examples

Colocá archivos CSV en `data/style_examples/` con columnas `question` y `answer`. Son ejemplos de cómo querés que se vean las respuestas (longitud, tono, formato). Se indexan automáticamente la primera vez que corrés `write`.

---

## Estructura del proyecto

```
company-profiler/
├── main.py                    # Entry point — 4 comandos
├── graph.py                   # Orquestador del pipeline
├── config.py                  # Settings desde .env
├── models.py                  # Pydantic models
│
├── agents/
│   ├── discovery_agent.py     # Busca empresas + evidence por requirement
│   ├── research_agent.py      # Crawl + extracción estructurada + FAISS
│   └── writer_agent.py        # Genera respuestas con RAG + Perplexity
│
├── rag/
│   ├── factual_rag.py         # Extracción estructurada + FAISS retrieval
│   └── style_rag.py           # Style examples retrieval
│
├── tools/
│   ├── crawling.py            # Spider, fetch, filtros de idioma/skip
│   ├── search.py              # Exa search + DuckDuckGo fallback
│   ├── perplexity.py          # Perplexity Agent API + Search API
│   ├── pdf_extractor.py       # Extracción de PDFs
│   ├── chunking.py            # Chunking de texto
│   ├── embeddings.py          # Embeddings locales (all-MiniLM-L6-v2)
│   ├── faiss_store.py         # FAISS index wrapper
│   ├── questions_loader.py    # Carga questions_example.csv
│   └── requirements_loader.py # Carga requirements_example.csv
│
├── data/
│   ├── style_examples/        # CSVs con ejemplos de estilo (question, answer)
│   ├── style_index.*          # FAISS index de style examples (auto-generado)
│   └── company_indexes/       # FAISS indexes por empresa (auto-generado)
│
├── output/                    # Todos los outputs con timestamp
├── cache/discovery/           # Cache de búsquedas de discovery por hash de requirements
│
├── requirements_example.csv
├── questions_example.csv
└── companies_list.csv
```

---

## Costos estimados de API

Por empresa investigada (con 57 preguntas):

| Componente | Costo aprox. |
|---|---|
| GPT-4.1-mini (extracción + writer) | ~$0.10 |
| Perplexity fast-search (12 batches × 5 preguntas) | ~$0.20 |
| Perplexity deep-research (fallback, ~5 preguntas sin evidencia) | ~$0.15 |
| Exa search | ~$0.02 |
| **Total por empresa** | **~$0.45–$0.55** |

Discovery (búsqueda inicial de empresas): ~$0.05–$0.10 total, se cachea por hash de requirements.

---

## Notas

- **Cache de discovery**: los resultados de búsqueda se cachean en `cache/discovery/` por hash MD5 de los requirements. Si no cambiás los requirements, el discovery usa el cache.
- **Deduplicación**: el discovery hace fuzzy matching de nombres (SequenceMatcher, threshold 0.85) para evitar empresas repetidas.
- **Filtrado de idiomas**: el crawler ignora automáticamente URLs con variantes de idioma (`/es/`, `/fr/`, `/chinese/`, etc.).
- **Embeddings locales**: se usa `all-MiniLM-L6-v2` via sentence-transformers, no requiere API key.
