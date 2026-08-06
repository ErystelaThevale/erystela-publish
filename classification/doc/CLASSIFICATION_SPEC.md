---
version: 1.0.0
updated: 2026-08-06
---

# Category Relationship Map — Spec

## Overview

Visualizes how Web of Science (WoS) subject categories relate to one another, based on
which categories are jointly assigned to the same underlying document ("theory"). Category
pairs that co-occur under the same document are treated as strongly connected.

- **Tool**: `src/app.py` (Streamlit)
- **Run**: `streamlit run src/app.py`
- **URL**: `http://localhost:8501` (default)

This repository ships a pre-generated snapshot of the assignment data (`data/classification.csv`)
so the tool runs standalone, without needing to regenerate that mapping yourself.

---

## Folder layout

```
classification/
├── data/
│   ├── categories.csv              # WoS category list (id, category)
│   ├── classification.csv          # document → category IDs (paper_id, category_ids)
│   └── wos_subject_categories.csv  # category → domain (subject_category, Domain-1, Domain-2)
├── output/
│   ├── classification_mapped.csv   # app.py auto-save (paper_id, category_names)
│   ├── category_edges.csv          # app.py auto-save (co-occurrence edge list)
│   └── category_graph.html         # app.py auto-save (the pyvis relationship map itself)
├── doc/
│   └── CLASSIFICATION_SPEC.md
└── src/
    └── app.py                      # relationship map GUI (Streamlit)
```

All paths in `app.py` are resolved relative to the script's own location
(`Path(__file__).resolve().parent`), so this folder is self-contained — it does not
assume any particular position inside a larger repository.

---

## Input

| File | Content | Required |
|---|---|---|
| `data/categories.csv` | Category ID → name mapping | Required (shows an error and stops if missing) |
| `data/classification.csv` | Document → category IDs mapping | Required (shows an error and stops if missing) |
| `data/wos_subject_categories.csv` | Category name → domain (up to 2), columns: `subject_category, Domain-1, Domain-2` | Optional (categories with no match fall back to "Unknown" / a neutral color) |

`data/classification.csv` is a wide-format CSV, header `paper_id, category_ids`, where
`category_ids` is a `;`-delimited list of category IDs (or `None` if none apply). How this
file is produced is out of scope for this tool — treat it as an input dataset.

---

## Co-occurrence computation (vectorized)

1. Build a document × category indicator matrix `M` (0/1, rows = documents, columns = categories in use)
2. Compute `cooc = M.T @ M`
   - **Diagonal** = number of documents referencing each category (drives node size)
   - **Off-diagonal** = co-occurrence count per category pair (drives edge width; pairs with 0 co-occurrences get no edge)

---

## Domain coloring

- Looks up `subject_category` in `wos_subject_categories.csv` by exact match on category name
- Fixed 6-domain palette (`DOMAIN_COLORS` in `app.py`; a dark-surface-validated categorical palette, CVD/contrast-checked for adjacent pairs)

  | Domain | Color |
  |---|---|
  | Arts & Humanities | `#3987e5` (blue) |
  | Clinical, Pre-Clinical & Health | `#d95926` (orange) |
  | Engineering & Technology | `#199e70` (aqua) |
  | Life Sciences | `#c98500` (yellow) |
  | Physical Sciences | `#d55181` (magenta) |
  | Social Sciences | `#008300` (green) |

- **Single domain**: uses that domain's color directly
- **Multiple domains (up to 2)**: gamma-corrected RGB blend of the two domain colors (`blend_hex()`)
- **Unknown domain** (no match in `wos_subject_categories.csv`): falls back to `DOMAIN_COLOR_DEFAULT` (`#898781`, muted gray)
- In all cases, the node's click-detail panel shows the original (pre-blend) domain name(s)

---

## Auto-save

Every time the page is opened, the following are recomputed and overwritten (no explicit
save action needed):

| File | Content |
|---|---|
| `output/classification_mapped.csv` | `paper_id, category_names` (IDs mapped to names, wide format) |
| `output/category_edges.csv` | `source, target, weight` (co-occurrence edge list, sorted by weight descending) |
| `output/category_graph.html` | The pyvis-generated relationship map itself (detail panel HTML/JS embedded) |

---

## GUI layout

Sidebar: metrics for document count / categories in use / edges (co-occurring pairs).

### Tab 1: Relationship Map

- **Document ID dropdown** (`(None selected)` + all document IDs): selecting one highlights that
  document's category nodes (white border + enlarged), dims all other nodes to 0.15 opacity.
  Edges are shaded in three tiers depending on whether both / one / neither endpoint is highlighted
- **Graph body** (5:1 column ratio with the legend): a force-directed network rendered by pyvis,
  embedded directly via `streamlit.components.v1.html`
- **Node click detail panel** (fixed, top-right): category name, domain (with color swatch),
  referencing-document count, and a scrollable list of related document IDs. Clicking the
  background resets it
  - vis.js's built-in tooltip (`title`) renders as plain text (HTML tags like `<br>` show up
    literally), so hover only shows the category name — details are rendered into a separate
    DOM panel on click instead
- **Domain legend** (right column): always-visible color swatches for all 6 domains

### Tab 2: Document–Category Table

Renders `classification_mapped.csv` as-is, in table form.

### Tab 3: Category Reference Ranking

Categories ranked by referencing-document count, descending.

---

## Custom node/edge attributes (vis.js DataSet)

In addition to pyvis's standard attributes, the following are attached
(retrievable via `network.body.data.nodes.get(id)`):

| Attribute | Content |
|---|---|
| `domain` | Domain name (`"A + B"` format if blended; pre-blend name) |
| `refcount` | Referencing-document count |
| `theories` | Sorted array of document IDs that use this category |

---

## Known constraints

- Matching against `wos_subject_categories.csv` is **exact name match only** — naming
  inconsistencies fall back to "Unknown" (neutral color)
- `category_graph.html` is overwritten on every run — save a copy elsewhere if you want to
  keep a snapshot
- `use_container_width` / `st.components.v1.html` are expected to be deprecated in a future
  Streamlit release (both still work as of the Streamlit version this was built against, with
  a deprecation warning)

---

## Required Python packages

```bash
pip install streamlit pandas numpy pyvis networkx
```
