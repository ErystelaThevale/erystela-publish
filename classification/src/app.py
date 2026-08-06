# -*- coding: utf-8 -*-
"""
カテゴリ関係マップ Streamlit ツール

- 入力: data/classification.csv （paper_id, category_ids）
        data/categories.csv （id, category）
        data/wos_subject_categories.csv （category, domain, ...）※任意
- 処理: 同一理論に紐づくカテゴリ同士を「結びつきが強い」とみなし、
        理論×カテゴリの指標行列（0/1）の内積（M^T @ M）で共起の重みを
        ベクトル計算する。対角成分 = カテゴリの被参照理論数（ノードサイズ）、
        非対角成分 = カテゴリ間の共起回数（エッジ太さ）。
- 出力（自動保存）:
        output/classification_mapped.csv （paper_id, category_names）
        output/category_edges.csv         （source, target, weight）
        output/category_graph.html         （pyvis 関係マップ）

使い方: streamlit run app.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── パス定数（ツール自身のフォルダ基準。Strategy側のディレクトリ構造には依存しない
#   ことで、classification/ フォルダごと他リポジトリへコピーしてもそのまま動く）──
TOOL_DIR   = Path(__file__).resolve().parent
DATA_DIR   = TOOL_DIR / "data"
OUTPUT_DIR = TOOL_DIR / "output"

CATEGORIES_CSV      = DATA_DIR / "categories.csv"
CLASSIFICATION_CSV  = DATA_DIR / "classification.csv"
DOMAINS_CSV         = DATA_DIR / "wos_subject_categories.csv"

MAPPED_CSV = OUTPUT_DIR / "classification_mapped.csv"
EDGES_CSV  = OUTPUT_DIR / "category_edges.csv"
GRAPH_HTML = OUTPUT_DIR / "category_graph.html"

DELIM      = ";"
NONE_TOKEN = "None"

# ドメイン配色（dataviz スキルの検証済みカテゴリカルパレット、dark面用の先頭6色を
# ドメイン名の辞書順で固定割当。node-validate_palette.js --mode dark --surface #1a1a2e
# で隣接ペア基準の CVD・コントラストを確認済み）
DOMAIN_COLORS = {
    "Arts & Humanities":               "#3987e5",  # blue
    "Clinical, Pre-Clinical & Health": "#d95926",  # orange
    "Engineering & Technology":        "#199e70",  # aqua
    "Life Sciences":                   "#c98500",  # yellow
    "Physical Sciences":               "#d55181",  # magenta
    "Social Sciences":                 "#008300",  # green
}
DOMAIN_COLOR_DEFAULT = "#898781"  # ドメイン不明時のフォールバック（muted）

st.set_page_config(page_title="Category Relationship Map", page_icon="🕸️", layout="wide")


@st.cache_data
def load_categories(path: Path) -> dict:
    """categories.csv を読み、{id: name} を返す。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # ヘッダ行を捨てる
        return {r[0].strip(): r[1].strip() for r in reader if len(r) >= 2 and r[0].strip()}


@st.cache_data
def load_domains(path: Path) -> dict:
    """wos_subject_categories.csv を読み、{category名: [domain, ...]} を返す（最大2件）。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # ヘッダ行を捨てる
        result = {}
        for r in reader:
            if not r or not r[0].strip():
                continue
            name = r[0].strip()
            domains = [d.strip() for d in r[1:3] if len(r) >= 2 and d.strip()]
            result[name] = domains
        return result


@st.cache_data
def load_classification(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def blend_hex(c1: str, c2: str) -> str:
    """2色をガンマ補正した上で平均し、知覚的に自然な中間色を返す。"""
    def to_linear(hexcode: str):
        rgb = [int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        return [c ** 2.2 for c in rgb]

    def to_srgb(linear):
        return [round((c ** (1 / 2.2)) * 255) for c in linear]

    a, b = to_linear(c1), to_linear(c2)
    mixed = [(x + y) / 2 for x, y in zip(a, b)]
    r, g, bch = to_srgb(mixed)
    return f"#{r:02x}{g:02x}{bch:02x}"


def category_color(name: str, domain_map: dict) -> tuple:
    """カテゴリ名からノード色とドメイン表示ラベルを決める。"""
    domains = domain_map.get(name, [])
    if not domains:
        return DOMAIN_COLOR_DEFAULT, "Unknown"
    if len(domains) == 1:
        return DOMAIN_COLORS.get(domains[0], DOMAIN_COLOR_DEFAULT), domains[0]
    c1 = DOMAIN_COLORS.get(domains[0], DOMAIN_COLOR_DEFAULT)
    c2 = DOMAIN_COLORS.get(domains[1], DOMAIN_COLOR_DEFAULT)
    return blend_hex(c1, c2), " + ".join(domains)


def build_paper_categories(df: pd.DataFrame) -> dict:
    """{paper_id: [category_id, ...]}。None・空欄は空リスト扱い。"""
    result = {}
    for _, row in df.iterrows():
        cell = row["category_ids"]
        if not cell or cell == NONE_TOKEN:
            result[row["paper_id"]] = []
        else:
            result[row["paper_id"]] = [c.strip() for c in cell.split(DELIM) if c.strip()]
    return result


def build_category_to_papers(paper_categories: dict) -> dict:
    """{category_id: [paper_id, ...]}（カテゴリ選択時に対応理論を表示するための逆引き）。"""
    result = {}
    for pid, cats in paper_categories.items():
        for cid in cats:
            result.setdefault(cid, []).append(pid)
    for cid in result:
        result[cid].sort()
    return result


def build_cooccurrence(paper_categories: dict):
    """理論×カテゴリの指標行列 M を作り、M^T @ M で共起行列を計算する。

    対角 = カテゴリの被参照理論数、非対角 = カテゴリペアの共起回数。
    """
    used_ids = sorted({cid for cats in paper_categories.values() for cid in cats})
    if not used_ids:
        return used_ids, np.zeros((0, 0), dtype=int)

    idx = {cid: i for i, cid in enumerate(used_ids)}
    M = np.zeros((len(paper_categories), len(used_ids)), dtype=int)
    for row, cats in enumerate(paper_categories.values()):
        for cid in cats:
            M[row, idx[cid]] = 1

    cooc = M.T @ M
    return used_ids, cooc


def build_network(
    used_ids, id_to_name, freq, edges_df: pd.DataFrame, domain_map: dict,
    category_to_papers: dict,
) -> Network:
    net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="#ffffff")
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -6000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.03
        },
        "minVelocity": 0.75
      },
      "edges": { "color": { "color": "#5a5a8a", "opacity": 0.5 }, "smooth": false },
      "nodes": { "font": { "size": 14 }, "borderWidth": 2, "shadow": true },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)

    max_freq = max(freq.values()) if freq else 1
    for cid in used_ids:
        n = freq.get(cid, 0)
        size = 10 + (n / max_freq) * 40
        name = id_to_name.get(cid, cid)
        color, domain_label = category_color(name, domain_map)
        theories = category_to_papers.get(cid, [])

        net.add_node(
            cid,
            label=name,
            size=size,
            title=name,  # ホバーは名前のみ。詳細はクリック時の別パネルに表示する
            color=color,
            domain=domain_label,
            refcount=n,
            theories=theories,
            basecolor=color,  # 理論ID選択時のハイライトJSが「元の色」に戻すための基準値
            basesize=size,
            opacity=1,
        )

    max_w = edges_df["weight"].max() if not edges_df.empty else 1
    for _, e in edges_df.iterrows():
        width = 1 + (e["weight"] / max_w) * 8
        net.add_edge(
            e["source_id"], e["target_id"],
            value=int(e["weight"]), width=width,
            title=f"Co-occurring theories: {e['weight']}",
        )

    return net


def make_detail_panel_html() -> str:
    """クリックしたノードの詳細（カテゴリ名・Domain・被参照数）を表示する別枠パネル。

    vis.js の node title はプレーンテキストとして描画され、<br> がそのまま
    文字列表示されてしまうため、ホバーではなくクリックで別 DOM を更新する
    方式に切り替えている。
    """
    return """
<div id="node-detail-panel" style="
  position: fixed;
  top: 20px;
  right: 20px;
  min-width: 220px;
  max-width: 300px;
  background: rgba(0,0,0,0.82);
  color: #fff;
  padding: 14px 18px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.8;
  z-index: 10;
">
  <div id="node-detail-empty" style="color:#aaa;">Click a category to see details</div>
  <div id="node-detail-content" style="display:none;">
    <div id="node-detail-name" style="font-size:15px; font-weight:bold; margin-bottom:6px;"></div>
    <div id="node-detail-domain"></div>
    <div id="node-detail-count"></div>
    <div id="node-detail-theories" style="
      margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.2);
      max-height:180px; overflow-y:auto; font-size:12px; color:#ddd;
    "></div>
  </div>
</div>

<script>
(function() {
  function waitForNetwork() {
    if (typeof network === "undefined") {
      setTimeout(waitForNetwork, 100);
      return;
    }
    setupDetailPanel();
  }

  function setupDetailPanel() {
    var emptyEl   = document.getElementById("node-detail-empty");
    var contentEl = document.getElementById("node-detail-content");
    var nameEl     = document.getElementById("node-detail-name");
    var domainEl   = document.getElementById("node-detail-domain");
    var countEl    = document.getElementById("node-detail-count");
    var theoriesEl = document.getElementById("node-detail-theories");

    network.on("click", function(params) {
      if (params.nodes.length === 0) {
        emptyEl.style.display = "block";
        contentEl.style.display = "none";
        return;
      }
      var node = network.body.data.nodes.get(params.nodes[0]);
      var swatch = (typeof node.color === "string") ? node.color
                 : (node.color && node.color.background) || "#fff";
      var theories = node.theories || [];

      nameEl.textContent = node.label;
      domainEl.innerHTML =
        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;' +
        'background:' + swatch + ';margin-right:6px;"></span>Domain: ' + node.domain;
      countEl.textContent = "Referencing theories: " + node.refcount;
      theoriesEl.innerHTML =
        'Related theories (' + theories.length + ')<br>' + theories.join(', ');

      emptyEl.style.display = "none";
      contentEl.style.display = "block";
    });
  }

  waitForNetwork();
})();
</script>
"""


def make_theory_selector_html(theory_ids: list) -> str:
    """理論IDで関連カテゴリを強調表示するセレクタ。

    ハイライト計算をクライアントサイドJSだけで行う（各ノードに埋め込み済みの
    `theories` から都度算出する）ため、サーバー（Streamlit）なしでも動く。
    これにより Streamlit 埋め込み時も GitHub Pages 単体表示時も、同じ
    category_graph.html がまったく同じように動作する。
    """
    no_selection = "(None selected)"
    options_json = json.dumps([no_selection] + theory_ids, ensure_ascii=False)

    template = """
<div id="theory-selector-panel" style="
  position: fixed;
  top: 20px;
  left: 20px;
  min-width: 260px;
  background: rgba(0,0,0,0.82);
  color: #fff;
  padding: 14px 18px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.6;
  z-index: 10;
">
  <label for="theory-select" style="display:block; margin-bottom:6px; font-weight:bold;">
    Highlight categories by theory ID
  </label>
  <select id="theory-select" style="width:100%; padding:4px; border-radius:4px;"></select>
  <div id="theory-caption" style="margin-top:8px; color:#ddd;"></div>
</div>

<script>
(function() {
  var NO_SELECTION = "__NO_SELECTION__";
  var DELIM = "__DELIM__";
  var THEORY_OPTIONS = __THEORY_OPTIONS_JSON__;

  var DEFAULT_EDGE_COLOR = { color: "#5a5a8a", opacity: 0.5 };
  var BOTH_EDGE_COLOR    = { color: "#ffffff", opacity: 0.9 };
  var EITHER_EDGE_COLOR  = { color: "#5a5a8a", opacity: 0.35 };
  var NEITHER_EDGE_COLOR = { color: "#5a5a8a", opacity: 0.05 };

  function waitForNetwork() {
    if (typeof network === "undefined") {
      setTimeout(waitForNetwork, 100);
      return;
    }
    setup();
  }

  function setup() {
    var select  = document.getElementById("theory-select");
    var caption = document.getElementById("theory-caption");

    THEORY_OPTIONS.forEach(function(id) {
      var opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      select.appendChild(opt);
    });

    select.addEventListener("change", function() {
      applySelection(select.value, caption);
    });
  }

  function applySelection(selected, caption) {
    var nodes = network.body.data.nodes;
    var edges = network.body.data.edges;

    if (selected === NO_SELECTION) {
      nodes.get().forEach(function(n) {
        nodes.update({ id: n.id, color: n.basecolor, size: n.basesize, borderWidth: 2, opacity: 1 });
      });
      edges.get().forEach(function(e) {
        edges.update({ id: e.id, color: DEFAULT_EDGE_COLOR });
      });
      caption.textContent = "";
      return;
    }

    var highlightIds = {};
    var names = [];
    nodes.get().forEach(function(n) {
      if ((n.theories || []).indexOf(selected) !== -1) {
        highlightIds[n.id] = true;
        names.push(n.label);
      }
    });

    nodes.get().forEach(function(n) {
      if (highlightIds[n.id]) {
        nodes.update({
          id: n.id,
          size: n.basesize + 8,
          color: { background: n.basecolor, border: "#ffffff", highlight: n.basecolor },
          borderWidth: 4,
          opacity: 1,
        });
      } else {
        nodes.update({ id: n.id, color: n.basecolor, size: n.basesize, borderWidth: 2, opacity: 0.15 });
      }
    });

    edges.get().forEach(function(e) {
      var both   = highlightIds[e.from] && highlightIds[e.to];
      var either = both || highlightIds[e.from] || highlightIds[e.to];
      var color  = both ? BOTH_EDGE_COLOR : (either ? EITHER_EDGE_COLOR : NEITHER_EDGE_COLOR);
      edges.update({ id: e.id, color: color });
    });

    if (names.length) {
      names.sort();
      caption.textContent = selected + " categories: " + names.join(DELIM);
    } else {
      caption.textContent = selected + " has no assigned categories.";
    }
  }

  waitForNetwork();
})();
</script>
"""
    return (
        template
        .replace("__NO_SELECTION__", no_selection)
        .replace("__DELIM__", DELIM)
        .replace("__THEORY_OPTIONS_JSON__", options_json)
    )


def main():
    st.title("🕸️ Category Relationship Map")

    if not CATEGORIES_CSV.exists():
        st.error(f"categories.csv not found: {CATEGORIES_CSV}")
        return
    if not CLASSIFICATION_CSV.exists():
        st.error(f"classification.csv not found: {CLASSIFICATION_CSV}")
        return

    id_to_name = load_categories(CATEGORIES_CSV)
    df = load_classification(CLASSIFICATION_CSV)
    paper_categories = build_paper_categories(df)
    category_to_papers = build_category_to_papers(paper_categories)

    domain_map = load_domains(DOMAINS_CSV) if DOMAINS_CSV.exists() else {}

    # ── 名称マッピング表 ──────────────────────────────────────────────
    mapped_rows = [
        {
            "paper_id": pid,
            "category_names": DELIM.join(id_to_name.get(c, c) for c in cats) if cats else NONE_TOKEN,
        }
        for pid, cats in paper_categories.items()
    ]
    mapped_df = pd.DataFrame(mapped_rows)

    # ── 共起（ベクトル計算） ──────────────────────────────────────────
    used_ids, cooc = build_cooccurrence(paper_categories)
    freq = {used_ids[i]: int(cooc[i, i]) for i in range(len(used_ids))}

    edges = [
        {
            "source_id": used_ids[i], "target_id": used_ids[j],
            "source": id_to_name.get(used_ids[i], used_ids[i]),
            "target": id_to_name.get(used_ids[j], used_ids[j]),
            "weight": int(cooc[i, j]),
        }
        for i in range(len(used_ids))
        for j in range(i + 1, len(used_ids))
        if cooc[i, j] > 0
    ]
    edges_df = (
        pd.DataFrame(edges).sort_values("weight", ascending=False)
        if edges else
        pd.DataFrame(columns=["source_id", "target_id", "source", "target", "weight"])
    )

    # ── 自動保存 ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(MAPPED_CSV, index=False, encoding="utf-8")
    edges_df.drop(columns=["source_id", "target_id"]).to_csv(EDGES_CSV, index=False, encoding="utf-8")

    with st.sidebar:
        st.metric("Theories", len(paper_categories))
        st.metric("Categories used", len(used_ids))
        st.metric("Edges (co-occurring pairs)", len(edges_df))

    tab_graph, tab_table, tab_freq = st.tabs(
        ["Relationship Map", "Theory-Category Table", "Category Reference Ranking"]
    )

    with tab_graph:
        if edges_df.empty:
            st.info("No co-occurring category pairs.")
        else:
            col_graph, col_legend = st.columns([5, 1])
            with col_graph:
                net = build_network(
                    used_ids, id_to_name, freq, edges_df, domain_map,
                    category_to_papers,
                )
                net.save_graph(str(GRAPH_HTML))
                html = GRAPH_HTML.read_text(encoding="utf-8")
                html = html.replace(
                    "</body>",
                    make_detail_panel_html()
                    + make_theory_selector_html(sorted(paper_categories.keys()))
                    + "</body>",
                )
                GRAPH_HTML.write_text(html, encoding="utf-8")
                components.html(html, height=800, scrolling=True)
                st.caption(f"Auto-saved: {GRAPH_HTML}")
            with col_legend:
                st.markdown("**Domain**")
                for domain, color in DOMAIN_COLORS.items():
                    st.markdown(
                        f'<span style="display:inline-block;width:12px;height:12px;'
                        f'border-radius:50%;background:{color};margin-right:6px;"></span>'
                        f'{domain}',
                        unsafe_allow_html=True,
                    )
                st.caption("Categories spanning multiple domains are shown with a blended color")

    with tab_table:
        st.dataframe(mapped_df, use_container_width=True, hide_index=True)
        st.caption(f"Auto-saved: {MAPPED_CSV}")

    with tab_freq:
        freq_df = pd.DataFrame(
            [{"category": id_to_name.get(cid, cid), "referencing_theories": n} for cid, n in freq.items()]
        ).sort_values("referencing_theories", ascending=False)
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        st.caption(f"Auto-saved: {EDGES_CSV}")


if __name__ == "__main__":
    main()
