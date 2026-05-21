# Graph Report - C:\Users\usuario\Downloads\TFG RAGI FINAL  (2026-04-23)

## Corpus Check
- Corpus is ~10,367 words - fits in a single context window. You may not need a graph.

## Summary
- 106 nodes · 188 edges · 7 communities detected
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Image Extraction|Image Extraction]]
- [[_COMMUNITY_Question Generation|Question Generation]]
- [[_COMMUNITY_Caption Description LLM|Caption Description LLM]]
- [[_COMMUNITY_LLM Descriptions|LLM Descriptions]]
- [[_COMMUNITY_Indexing Pipeline|Indexing Pipeline]]
- [[_COMMUNITY_Backend API|Backend API]]
- [[_COMMUNITY_Search Engine|Search Engine]]

## God Nodes (most connected - your core abstractions)
1. `generate_questions()` - 14 edges
2. `generate_desc3()` - 12 edges
3. `generate_good_desc()` - 11 edges
4. `search()` - 10 edges
5. `call_ollama()` - 6 edges
6. `main()` - 6 edges
7. `call_ollama()` - 5 edges
8. `fallback_general_questions()` - 5 edges
9. `main()` - 5 edges
10. `panel_prefix()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `looks_spanish()` --calls--> `search()`  [INFERRED]
  C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\02_descripciones_llm.py → C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\backend.py
- `mentions_other_figures()` --calls--> `search()`  [INFERRED]
  C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\02_descripciones_llm.py → C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\backend.py
- `looks_generic()` --calls--> `search()`  [INFERRED]
  C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\02_descripciones_llm.py → C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\backend.py
- `mentions_other_figures()` --calls--> `search()`  [INFERRED]
  C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\03_descripciones_llm_caption.py → C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\backend.py
- `looks_generic()` --calls--> `search()`  [INFERRED]
  C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\03_descripciones_llm_caption.py → C:\Users\usuario\Downloads\TFG RAGI FINAL\scripts\backend.py

## Hyperedges (group relationships)
- **RAG Pipeline** — 01_extraccion_imagenes_py, 02_descripciones_llm_py, 05_indexacion_py, 06_buscador_py [INFERRED 0.85]

## Communities

### Community 0 - "Image Extraction"
Cohesion: 0.12
Nodes (14): box_iou(), clamp(), clean_text(), compute_barriers(), dedup_keep_best(), drop_contained_boxes(), expand_by_content(), expand_with_small_plaintext_labels() (+6 more)

### Community 1 - "Question Generation"
Cohesion: 0.19
Nodes (22): bad_question(), build_fix_prompt(), build_prompt(), call_ollama(), clean_text(), estimate_panels(), extract_anchor_terms(), fallback_general_questions() (+14 more)

### Community 2 - "Caption Description LLM"
Cohesion: 0.23
Nodes (17): build_fix_prompt(), build_prompt(), call_ollama(), clean_llm_output(), contradicts_panels(), estimate_panels(), generate_desc3(), is_one_sentence() (+9 more)

### Community 3 - "LLM Descriptions"
Cohesion: 0.27
Nodes (14): build_fix_prompt(), build_prompt(), clean_llm_output(), generate_good_desc(), infer_hint_from_caption(), is_one_sentence(), looks_generic(), looks_spanish() (+6 more)

### Community 4 - "Indexing Pipeline"
Cohesion: 0.29
Nodes (9): build_full_text(), create_index(), get_embedding(), index_document(), main(), Indexa un documento en ElasticSearch., Crea el índice con mapping para búsqueda semántica (kNN)., Combina caption + description2 + description3 + questions en un solo texto. (+1 more)

### Community 5 - "Backend API"
Cohesion: 0.24
Nodes (5): get_embedding(), RatingRequest, search(), SearchRequest, BaseModel

### Community 6 - "Search Engine"
Cohesion: 0.53
Nodes (5): get_embedding(), main(), print_results(), Dado un texto de consulta, devuelve las top_k imágenes más relevantes., search()

## Knowledge Gaps
- **5 isolated node(s):** `Crea el índice con mapping para búsqueda semántica (kNN).`, `Combina caption + description2 + description3 + questions en un solo texto.`, `Llama al modelo de embeddings del servidor de la universidad o local.`, `Indexa un documento en ElasticSearch.`, `Dado un texto de consulta, devuelve las top_k imágenes más relevantes.`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `search()` connect `Backend API` to `Question Generation`, `Caption Description LLM`, `LLM Descriptions`?**
  _High betweenness centrality (0.735) - this node is a cross-community bridge._
- **Why does `parse_questions_json()` connect `Question Generation` to `Backend API`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `bad_question()` connect `Question Generation` to `Backend API`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `search()` (e.g. with `looks_spanish()` and `mentions_other_figures()`) actually correct?**
  _`search()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Crea el índice con mapping para búsqueda semántica (kNN).`, `Combina caption + description2 + description3 + questions en un solo texto.`, `Llama al modelo de embeddings del servidor de la universidad o local.` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Image Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._