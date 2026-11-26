# `mcp-gaviao-real`

Servidor **MCP (Model Context Protocol)** dedicado à extração inteligente de documentos (**PDF, DOCX, PPTX, XLSX, HTML, CSV**) para LLMs locais e em nuvem, com **roteamento híbrido por página**, **sistema automático de adaptação ao hardware**, **cache granular SQLite + FTS5** e **pipeline visual com carregamento sob demanda e zoom 300 DPI**.

---

## Principais Recursos

- **Estratégia Híbrida por Página:** analisa cada página do PDF em `< 5 ms` (`triage.py`) e escolhe o motor ideal:
  - **`pymupdf_fast`** (`pymupdf` / `pymupdf4llm`): extração instantânea em CPU para PDFs digitais limpos;
  - **`markitdown_fast`** (`markitdown` + `openpyxl` + `python-pptx`): extração direta para DOCX, PPTX, XLSX (paginado por aba), HTML e CSV;
  - **`docling_layout` / `docling_ocr`** (`docling` v2 + `rapidocr-onnxruntime`, extra opcional): layout multi-coluna, `TableFormer` com suporte a células mescladas e OCR completo para páginas escaneadas ou com fontes `ToUnicode` corrompidas (`\ufffd`).
- **Adaptação Automática à Máquina (`hardware.py`):** detecta SO, núcleos, RAM e GPU/VRAM (NVIDIA CUDA ou Apple Silicon MPS) sem importar `torch` no arranque, ajustando batch size, threads, limite de páginas OCR e descarregamento de VRAM.
- **Pipeline Visual "Olho de Gavião":**
  - Filtra ícones pequenos, linhas e **logos repetidos** em cabeçalhos/rodapés (por `xref` e *dHash* perceptual em 3+ páginas);
  - Captura gráficos e diagramas vetoriais via *clip* renderizado (`cluster_drawings`);
  - Entrega imagens em **WebP** redimensionadas (`<= 1280 px`), renderização de página inteira (`page:N`) e **zoom 300 DPI** via `crop_bbox=[x0, y0, x1, y1]`;
  - Fallback opcional para **auto-legenda via VLM** OpenAI-compatível (`vlm_captions=True`).
- **Cache Granular + Busca Híbrida FTS5:** armazena páginas extraídas e blobs WebP em diretório local do sistema (`platformdirs`), com busca **BM25 insensível a acentos** (`unicode61 remove_diacritics 2`) + Regex e poda automática LRU.

---

## Sistema de Adaptação à Máquina

| Perfil (`HardwareTier`) | Critério de Detecção | Comportamento Automático |
| --- | --- | --- |
| **`gpu_high`** | NVIDIA CUDA `>= 8 GB` VRAM + `docling` | `device="cuda"`, `batch_size=4`, descarrega VRAM após 300 s ocioso, escala layouts complexos e OCR para GPU |
| **`gpu_low`** | CUDA `3.5–7.9 GB` ou Apple `mps` + `docling` | `batch_size=1`, limpa VRAM imediatamente após cada job, máx. 5 páginas OCR por chamada |
| **`cpu_rich`** | Sem GPU, RAM `>= 8 GB` + `docling` | `device="cpu"`, `omp_threads` calibrado, máx. 3 páginas OCR por chamada |
| **`lite`** | Sem `docling` ou RAM `< 8 GB` | Roda exclusivamente com `PyMuPDF` + `MarkItDown`; páginas escaneadas retornam aviso estruturado sugerindo `get_document_image(target="page:N")` |

---

## Instalação Portátil (via `uv`)

```powershell
# 1. Instalação base leve (CPU universal: PyMuPDF + MarkItDown)
uv sync

# 2. Ou deixe o assistente detectar sua máquina e aplicar o perfil ideal:
uv run mcp-gaviao-real setup
uv run mcp-gaviao-real setup --apply
```

Instalação manual por perfil de máquina:
- **NVIDIA CUDA:** `uv sync --extra accurate --extra cuda`
- **Apple Silicon (MPS):** `uv sync --extra accurate`
- **CPU-only (`>= 12 GB` RAM):** `uv sync --extra accurate --extra cpu`
- **Desenvolvimento e testes:** `uv sync --extra dev`

Verifique o ambiente a qualquer momento com:

```powershell
uv run mcp-gaviao-real doctor
```

---

## Tools e Resources MCP

Todas as ferramentas possuem `ToolAnnotations(read_only_hint=True, idempotent_hint=True)`:

1. **`inspect_document(source)`**: retorna `doc_id`, perfil ativo, TOC, estimativa de tokens, classificação por página e inventário de imagens sem carregar modelos pesados.
2. **`read_document(source, pages="1-5", mode="auto", include_images=False, vlm_captions=False, max_tokens=12000)`**: extrai Markdown limpo com *breadcrumbs* de seção, placeholders `doc://{doc_id}/img/{id}` e proteção contra estouro de contexto.
3. **`get_document_image(source, target, max_size=1280, crop_bbox=None)`**: retorna `ImageContent` WebP para uma figura (`target="p2_fig1"`), página inteira (`target="page:3"`) ou **zoom 300 DPI** (`target="page:1", crop_bbox=[510, 200, 960, 330]` na escala `0..1000`).
4. **`search_document(source, query, mode="hybrid", limit=10)`**: busca híbrida (`FTS5 BM25` sem acentos + Regex) retornando páginas, *breadcrumbs*, trechos e `unindexed_scanned_pages`.

**MCP Resource Templates:**
- `doc://{doc_id}/page/{page_num}` (`text/markdown`)
- `doc://{doc_id}/img/{image_id}` (`image/webp`)

---

## Configuração no Cliente MCP

Gere o bloco pronto para o seu `mcp_config.json`:

```powershell
# Modo local stdio (recomendado)
uv run mcp-gaviao-real setup --print-mcp-config

# Modo servidor HTTP (Streamable HTTP)
uv run mcp-gaviao-real setup --print-mcp-config --transport streamable-http --port 8765
uv run mcp-gaviao-real serve --transport streamable-http --host 127.0.0.1 --port 8765
```

---

## Variáveis de Ambiente (`GAVIAO_*`)

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `GAVIAO_PROFILE` | `auto` | Força `auto`, `gpu_high`, `gpu_low`, `cpu_rich` ou `lite` |
| `GAVIAO_DEVICE` | `auto` | Força `auto`, `cuda`, `mps` ou `cpu` |
| `GAVIAO_CACHE_DIR` | `user_cache_dir` | Diretório do SQLite, blobs WebP e `hf_cache` |
| `GAVIAO_MAX_CACHE_BYTES` | `2147483648` | Limite LRU do cache em bytes (2 GiB) |
| `GAVIAO_ALLOWED_DIRS` | `[]` | Lista JSON opcional de diretórios permitidos para leitura |
| `GAVIAO_OCR_LANGS` | `["pt", "en"]` | Idiomas prioritários do OCR |
| `GAVIAO_VLM_BASE_URL` | `None` | Endpoint OpenAI-compatível para `vlm_captions=True` (Ollama, LM Studio, Gemini) |
| `GAVIAO_VLM_MODEL` | `None` | Nome do modelo VLM |
| `GAVIAO_VLM_API_KEY` | `None` | Chave opcional para o endpoint VLM |

---

## Testes

```powershell
uv run pytest -v
```
