#!/usr/bin/env python3
"""Generate benchmarks/skribe_inspector.html from cache files.

Usage:
    python benchmarks/build_skribe_inspector.py
    python benchmarks/build_skribe_inspector.py --cache-dir path/to/cache --output path/to/out.html

The script reads every JSON file produced by run_openml_fit.py (via
run_all_models.sh), picks the best-accuracy run per (dataset, model) pair, and
writes a self-contained HTML page with filterable model/dataset chips,
percentage bars with baseline reference lines, and collapsible prompt +
generated-code panels with syntax highlighting.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from skribe.prompt_markers import CONTEXT_END, CONTEXT_START, DATA_MARKER

# ---------------------------------------------------------------------------
# Model display config — extend here when new models are added to the bench
# ---------------------------------------------------------------------------

MODEL_META = {
    # OpenAI — base
    "gpt-4o":           {"label": "GPT-4o",            "color": "#d97575"},
    "gpt-4o-mini":      {"label": "GPT-4o mini",       "color": "#c55f5f"},
    "gpt-4.1":          {"label": "GPT-4.1",           "color": "#b54848"},
    "gpt-5.4-mini":     {"label": "GPT-5.4 mini",      "color": "#a03030"},
    "gpt-5.5":          {"label": "GPT-5.5",           "color": "#8b1a1a"},
    # OpenAI — +web
    "gpt-4o-mini+web":  {"label": "GPT-4o mini +web",  "color": "#e8a0a0"},
    "gpt-4.1+web":      {"label": "GPT-4.1 +web",      "color": "#d98888"},
    "gpt-5.4-mini+web": {"label": "GPT-5.4 mini +web", "color": "#c87070"},
    "gpt-5.5+web":      {"label": "GPT-5.5 +web",      "color": "#b85858"},
    # Google — base
    "vertex_ai/gemini-2.5-flash":      {"label": "Gemini 2.5 Flash",      "color": "#4285f4"},
    "vertex_ai/gemini-2.5-pro":        {"label": "Gemini 2.5 Pro",        "color": "#2c6fd4"},
    "vertex_ai/gemini-2.5-flash-lite": {"label": "Gemini 2.5 Flash Lite", "color": "#5b9cf5"},
    "vertex_ai/gemini-3.5-flash":      {"label": "Gemini 3.5 Flash",      "color": "#1a56c4"},
    # Google — +web
    "vertex_ai/gemini-2.5-flash+web":      {"label": "Gemini 2.5 Flash +web",      "color": "#80b0fa"},
    "vertex_ai/gemini-2.5-pro+web":        {"label": "Gemini 2.5 Pro +web",        "color": "#6a9de8"},
    "vertex_ai/gemini-2.5-flash-lite+web": {"label": "Gemini 2.5 Flash Lite +web", "color": "#90bffc"},
    "vertex_ai/gemini-3.5-flash+web":      {"label": "Gemini 3.5 Flash +web",      "color": "#5080d8"},
}

BASELINE_META = {
    "logreg": {"label": "LogReg", "color": "#9c27b0"},
    "xgboost": {"label": "XGBoost", "color": "#ff6f00"},
    "tabpfn": {"label": "TabPFN", "color": "#00838f"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cache(cache_dir: Path) -> tuple[dict, dict]:
    """Return (baselines, results) dicts.

    baselines: {dataset: {metric_name: accuracy}}
    results:   {(dataset, model_id): {accuracy, fit_prompt, generated_code, fit_time_s}}
    """
    baselines: dict = {}
    results: dict = {}

    for path in sorted(cache_dir.glob("*.json")):
        data = json.loads(path.read_text())

        model_id = data.get("model_id", "")
        dataset = data.get("dataset", "")
        if not model_id or not dataset:
            continue

        if model_id in BASELINE_META:
            # filename: <dataset>-<learner>-<hash>.json; metrics live under
            # data[model_id] (e.g. data["logreg"]["accuracy"]).
            acc = data.get(model_id, {}).get("accuracy")
            if acc is not None:
                baselines.setdefault(dataset, {})[model_id] = acc
            continue

        pl = data.get("skribe", {})
        if pl.get("error") or pl.get("accuracy") is None:
            continue

        key = (dataset, model_id)
        acc = pl["accuracy"]
        if key not in results or acc > results[key]["accuracy"]:
            results[key] = {
                "accuracy": acc,
                "fit_prompt": pl.get("fit_prompt") or "",
                "generated_code": pl.get("generated_code") or "",
                "fit_time_s": pl.get("fit_time_s") or 0,
                "prepass_time_s": pl.get("prepass_time_s") or 0,
                "predict_time_s": pl.get("predict_time_s") or 0,
                "context_prepass_prompt": pl.get("context_prepass_prompt") or "",
                "context_summary": pl.get("context_summary") or "",
            }

    return baselines, results


def build_groups(baselines: dict, results: dict) -> list[dict]:
    """Assemble per-dataset groups sorted by dataset name, models best-to-worst by accuracy."""
    datasets = sorted({ds for ds, _ in results})

    groups = []
    for dataset in datasets:
        models = []
        for model_id in MODEL_META:
            key = (dataset, model_id)
            if key not in results:
                continue
            r = results[key]
            models.append(
                {
                    "model_id": model_id,
                    "accuracy": round(r["accuracy"], 6),
                    "fit_prompt": r["fit_prompt"],
                    "generated_code": r["generated_code"],
                    "fit_time_s": round(r["fit_time_s"], 2),
                    "prepass_time_s": round(r["prepass_time_s"], 2),
                    "predict_time_s": round(r["predict_time_s"], 4),
                    "context_prepass_prompt": r["context_prepass_prompt"],
                    "context_summary": r["context_summary"],
                }
            )
        models.sort(key=lambda m: m["accuracy"], reverse=True)
        if not models:
            continue
        groups.append(
            {
                "dataset": dataset,
                "models": models,
                "baselines": baselines.get(dataset, {}),
            }
        )
    return groups


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

CSS = textwrap.dedent("""\
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f5f7; color: #1a1a1a; }

    header { background: #16213e; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.3); }
    header h1 { font-size: 1.15rem; font-weight: 700; }
    header p { font-size: 0.8rem; opacity: 0.6; margin-top: 2px; }

    .filter-bar { background: white; border-bottom: 1px solid #e0e0e0; padding: 10px 24px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; position: sticky; top: 56px; z-index: 99; }
    .filter-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #888; white-space: nowrap; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { padding: 4px 12px; border-radius: 99px; font-size: 0.78rem; cursor: pointer; border: 1.5px solid #ddd; background: white; color: #555; transition: all .15s; user-select: none; }
    .chip:hover { border-color: #aaa; }
    .chip.active { color: white; border-color: transparent; }

    main { max-width: 100%; margin: 0; padding: 24px; display: flex; flex-direction: column; gap: 32px; }

    .ds-group { background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }
    .ds-header { padding: 16px 20px; background: #f8f9fa; border-bottom: 1px solid #e8e8e8; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
    .ds-name { font-size: 1.1rem; font-weight: 700; }

    .baselines { display: flex; gap: 20px; flex-wrap: wrap; margin-left: auto; }
    .bl-pill { display: flex; align-items: center; gap: 6px; background: #f0f0f0; border-radius: 6px; padding: 5px 14px; font-size: 0.75rem; }
    .bl-pill .bl-name { color: #666; font-weight: 600; }
    .bl-pill .bl-acc { font-weight: 700; color: #333; }

    /* model-rows: grid columns = model-tag | bar | acc% | fit-time | toggle */
    .model-rows { display: flex; flex-direction: column; }
    .model-row { border-top: 1px solid #f0f0f0; }
    .model-row-header {
      position: relative;
      padding: 10px 24px;
      display: grid;
      grid-template-columns: 190px 1fr 52px 60px 20px;
      align-items: center;
      gap: 12px;
      cursor: pointer;
      transition: background .1s;
      overflow: hidden;
    }
    .full-ref-line {
      position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px;
      opacity: 0.25; pointer-events: none;
    }
    .model-row-header:hover { background: #fafafa; }
    .model-tag { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.75rem; font-weight: 700; color: white; white-space: nowrap; justify-self: start; }
    .acc-bar-track { position: relative; width: 100%; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
    .acc-bar { height: 100%; border-radius: 4px; }
    .bar-legend {
      padding: 8px 24px 2px;
      display: grid;
      grid-template-columns: 190px 1fr 52px 60px 20px;
      gap: 12px;
      background: #fafafa;
      border-top: 1px solid #f0f0f0;
    }
    .bar-legend-track { position: relative; height: 62px; }
    .ref-tick-label {
      position: absolute; bottom: 0; font-size: 0.62rem; font-weight: 700; white-space: nowrap;
      transform: translateX(-50%);
      padding: 1px 4px; border-radius: 3px; color: white;
    }
    .acc-text { font-size: 0.85rem; font-weight: 700; text-align: right; }
    .fit-time { font-size: 0.75rem; color: #aaa; text-align: right; white-space: nowrap; }
    .timing-bar { display: flex; gap: 6px; align-items: center; margin: 12px 0 4px; flex-wrap: wrap; }
    .timing-segment { display: flex; align-items: center; gap: 5px; font-size: 0.75rem; }
    .timing-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
    .timing-label { color: #888; }
    .timing-value { font-weight: 700; color: #444; font-family: "SF Mono","Fira Code",monospace; }
    .timing-divider { color: #ddd; }
    .toggle-icon { font-size: 0.75rem; color: #aaa; text-align: center; transition: transform .2s; }
    .toggle-icon.open { transform: rotate(90deg); }

    .model-detail { display: none; padding: 0 20px 20px; background: #fafcff; border-top: 1px solid #eef2ff; }
    .model-detail.open { display: block; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
    @media (max-width: 860px) { .detail-grid { grid-template-columns: 1fr; } }

    .pane-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: #888; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    .copy-btn { font-size: 0.7rem; text-transform: none; letter-spacing: 0; background: #eee; border: none; border-radius: 4px; padding: 2px 8px; cursor: pointer; color: #555; font-family: inherit; }
    .copy-btn:hover { background: #ddd; }
    .copy-btn:active { background: #ccc; }

    .prompt-box { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px; font-family: "SF Mono","Fira Code",monospace; font-size: 0.76rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-height: 440px; overflow-y: auto; color: #2d2d2d; }
.prompt-section-label { color: #1565c0; font-weight: 700; }

    pre { margin: 0; border-radius: 8px; overflow: auto; max-height: 440px; border: 1px solid #e0e0e0; min-width: 0; width: 100%; }
    .detail-grid > div { min-width: 0; }
    pre code { font-size: 0.76rem !important; line-height: 1.5 !important; }
    .no-data { color: #aaa; font-style: italic; font-size: 0.85rem; padding: 20px 0; }
""")

JS = textwrap.dedent("""\
    function esc(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    function accColor(a, logreg) {
      if (a === null || a === undefined) return "#ccc";
      const threshold = (logreg !== null && logreg !== undefined) ? logreg : 0.75;
      if (a >= threshold) {
        if (a >= 0.95) return "#1b5e20";
        if (a >= 0.90) return "#2e7d32";
        if (a >= 0.80) return "#558b2f";
        return "#7cb342";
      }
      return a >= threshold * 0.93 ? "#f57c00" : "#c62828";
    }

    function renderPrompt(text) {
      if (!text) return "<span class='no-data'>not stored</span>";
      const lines = text.split("\\n");
      let out = "", emitted = false;
      for (const line of lines) {
        if (!emitted && line.trim() === "") continue;
        firstLine = false; emitted = true;
        if (line === "__CONTEXT_START__")
          out += `<span class="prompt-section-label">── Dataset context ──</span>\\n`;
        else if (line === "__CONTEXT_END__")
          out += `<span class="prompt-section-label">── End context ──</span>\\n`;
        else if (line.trim().startsWith("Output a single valid Python"))
          out += `<span class="prompt-section-label">── Task instructions ──</span>\\n${esc(line)}\\n`;
        else if (line.trim() === "__DATA_MARKER__")
          out += `\\n<span class="prompt-section-label">── Training data ──</span>\\n`;
        else
          out += esc(line) + "\\n";
      }
      return out;
    }

    function makeDetailId(ds, mid) {
      return "detail-" + (ds + "-" + mid).replace(/[^a-z0-9]/gi, "_");
    }

    function renderGroups() {
      const main = document.getElementById("main");
      main.innerHTML = "";

      for (const group of GROUPS) {
        if (!activeDatasets.has(group.dataset)) continue;
        const visModels = group.models.filter(m => activeModels.has(m.model_id));
        if (!visModels.length) continue;

        const card = document.createElement("div");
        card.className = "ds-group";

        const blPills = Object.entries(BL_LABELS).map(([k, label]) => {
          const acc = group.baselines[k];
          if (acc == null) return "";
          return `<div class="bl-pill">
            <span class="bl-name" style="color:${BL_COLORS[k]}">${label}</span>
            <span class="bl-acc">${(acc*100).toFixed(1)}%</span>
          </div>`;
        }).join("");

        card.innerHTML = `
          <div class="ds-header">
            <span class="ds-name">${esc(group.dataset)}</span>
            <div class="baselines">${blPills}</div>
          </div>
          <div class="model-rows" id="rows-${esc(group.dataset)}"></div>`;

        const rowsEl = card.querySelector(".model-rows");

        // Legend row: baseline label pills staggered vertically to avoid overlap
        const blEntries = Object.entries(BL_LABELS)
          .map(([k, label]) => ({ k, label, acc: group.baselines[k] }))
          .filter(e => e.acc != null)
          .sort((a, b) => a.acc - b.acc);
        const LABEL_H = 16, LABEL_GAP = 4;
        const legendTicks = blEntries.map((e, i) => {
          const labelBottom = i * (LABEL_H + LABEL_GAP);
          return `<div class="ref-tick-label" style="left:${(e.acc*100).toFixed(2)}%;bottom:${labelBottom}px;background:${BL_COLORS[e.k]}">${e.label} ${(e.acc*100).toFixed(0)}%</div>`;
        }).join("");
        rowsEl.insertAdjacentHTML("beforebegin", `
          <div class="bar-legend">
            <div></div>
            <div class="bar-legend-track">${legendTicks}</div>
            <div></div><div></div><div></div>
          </div>`);

        for (const m of visModels) {
          const detailId = makeDetailId(group.dataset, m.model_id);
          const acc = m.accuracy;
          const color = accColor(acc, group.baselines.logreg);
          const row = document.createElement("div");
          row.className = "model-row";
          row.innerHTML = `
            <div class="model-row-header" onclick="toggleDetail('${detailId}', this)">
              <span class="model-tag" style="background:${MODEL_COLORS[m.model_id]}">${MODEL_LABELS[m.model_id]}</span>
              <div class="acc-bar-track">
                <div class="acc-bar" style="width:${(acc*100).toFixed(1)}%;background:${color}"></div>
              </div>
              <span class="acc-text" style="color:${color}">${(acc*100).toFixed(1)}%</span>
              <span class="fit-time">${m.fit_time_s ? m.fit_time_s + "s" : ""}</span>
              <span class="toggle-icon">▶</span>
            </div>
            <div class="model-detail" id="${detailId}">
              ${(() => {
                const parts = [];
                if (m.prepass_time_s) parts.push(`<span class="timing-segment"><span class="timing-swatch" style="background:#7b61ff"></span><span class="timing-label">pre-pass</span> <span class="timing-value">${m.prepass_time_s}s</span></span><span class="timing-divider">·</span>`);
                const fit_excl = m.prepass_time_s ? Math.max(0, m.fit_time_s - m.prepass_time_s).toFixed(2) : m.fit_time_s;
                parts.push(`<span class="timing-segment"><span class="timing-swatch" style="background:#1976d2"></span><span class="timing-label">fit</span> <span class="timing-value">${fit_excl}s</span></span>`);
                if (m.predict_time_s) parts.push(`<span class="timing-divider">·</span><span class="timing-segment"><span class="timing-swatch" style="background:#2e7d32"></span><span class="timing-label">predict</span> <span class="timing-value">${(m.predict_time_s * 1000).toFixed(1)}ms</span></span>`);
                return parts.length ? `<div class="timing-bar">${parts.join('')}</div>` : '';
              })()}
              <div class="detail-grid">
                ${m.context_prepass_prompt ? `
                <div>
                  <div class="pane-title">Context pre-pass prompt
                    <button class="copy-btn" onclick="copyId('prepass-prompt-${detailId}');event.stopPropagation()">copy</button>
                  </div>
                  <div class="prompt-box" id="prepass-prompt-${detailId}">${renderPrompt(m.context_prepass_prompt)}</div>
                </div>
                <div>
                  <div class="pane-title">Context summary (LLM output)
                    <button class="copy-btn" onclick="copyId('prepass-summary-${detailId}');event.stopPropagation()">copy</button>
                  </div>
                  <div class="prompt-box" id="prepass-summary-${detailId}">${m.context_summary ? renderPrompt(m.context_summary) : '<span style="color:#aaa">—</span>'}</div>
                </div>` : ''}
                <div>
                  <div class="pane-title">Fit prompt (sent to LLM)
                    <button class="copy-btn" onclick="copyId('prompt-${detailId}');event.stopPropagation()">copy</button>
                  </div>
                  <div class="prompt-box" id="prompt-${detailId}">${renderPrompt(m.fit_prompt)}</div>
                </div>
                <div>
                  <div class="pane-title">Generated Python
                    <button class="copy-btn" onclick="copyId('code-${detailId}');event.stopPropagation()">copy</button>
                  </div>
                  ${m.generated_code
                    ? `<pre><code class="language-python" id="code-${detailId}">${esc(m.generated_code)}</code></pre>`
                    : `<div class="no-data">generated_code not stored</div>`}
                </div>
              </div>
            </div>`;

          // Baseline reference lines injected into each header so they're clipped
          // to the header row and never overlap the expanded detail panel.
          // Bar column: 24px pad + 190px tag + 12px gap = 226px from left;
          //             24px pad + 20px toggle + 12px + 60px time + 12px + 52px acc + 12px = 192px from right.
          const headerEl = row.querySelector(".model-row-header");
          Object.entries(BL_LABELS).forEach(([k, label]) => {
            const bAcc = group.baselines[k];
            if (bAcc == null) return;
            const line = document.createElement("div");
            line.className = "full-ref-line";
            line.style.left = `calc(226px + ${bAcc} * (100% - 418px))`;
            line.style.background = BL_COLORS[k];
            line.title = `${label}: ${(bAcc*100).toFixed(1)}%`;
            headerEl.appendChild(line);
          });

          rowsEl.appendChild(row);
        }

        main.appendChild(card);
      }
    }

    function toggleDetail(id, header) {
      const el = document.getElementById(id);
      const icon = header.querySelector(".toggle-icon");
      const wasOpen = el.classList.contains("open");
      el.classList.toggle("open", !wasOpen);
      icon.classList.toggle("open", !wasOpen);
      if (!wasOpen) hljs.highlightAll();
    }

    function copyId(id) {
      const el = document.getElementById(id);
      const text = el.innerText || el.textContent || "";
      // Try modern clipboard API first (requires HTTPS or localhost)
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => flashBtn(id)).catch(() => legacyCopy(text, id));
      } else {
        legacyCopy(text, id);
      }
    }

    function legacyCopy(text, id) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;top:0;left:0;opacity:0;";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try { document.execCommand("copy"); flashBtn(id); } catch(e) {}
      document.body.removeChild(ta);
    }

    function flashBtn(id) {
      // Find the copy button associated with this pane and briefly turn it green
      const btn = document.querySelector(`[onclick*="'${id}'"]`);
      if (!btn) return;
      const orig = btn.textContent;
      btn.textContent = "✓ copied";
      btn.style.background = "#c8e6c9";
      setTimeout(() => { btn.textContent = orig; btn.style.background = ""; }, 1200);
    }

    // Model filter chips
    Object.entries(MODEL_LABELS).forEach(([mid, label]) => {
      const chip = document.createElement("div");
      chip.className = "chip active";
      chip.style.cssText = `background:${MODEL_COLORS[mid]};border-color:${MODEL_COLORS[mid]}`;
      chip.textContent = label;
      chip.addEventListener("click", () => {
        const on = activeModels.has(mid);
        on ? activeModels.delete(mid) : activeModels.add(mid);
        chip.classList.toggle("active", !on);
        chip.style.background = on ? "white" : MODEL_COLORS[mid];
        chip.style.color      = on ? MODEL_COLORS[mid] : "white";
        chip.style.borderColor = on ? MODEL_COLORS[mid] : MODEL_COLORS[mid];
        renderGroups();
      });
      document.getElementById("model-filters").appendChild(chip);
    });

    // Dataset filter chips
    GROUPS.forEach(g => {
      const chip = document.createElement("div");
      chip.className = "chip active";
      chip.style.cssText = "background:#444;border-color:#444;color:white";
      chip.textContent = g.dataset;
      chip.addEventListener("click", () => {
        const on = activeDatasets.has(g.dataset);
        on ? activeDatasets.delete(g.dataset) : activeDatasets.add(g.dataset);
        chip.classList.toggle("active", !on);
        chip.style.background  = on ? "white" : "#444";
        chip.style.color       = on ? "#444"  : "white";
        chip.style.borderColor = on ? "#ddd"  : "#444";
        renderGroups();
      });
      document.getElementById("dataset-filters").appendChild(chip);
    });

    renderGroups();
""").replace("__CONTEXT_START__", CONTEXT_START).replace("__CONTEXT_END__", CONTEXT_END).replace("__DATA_MARKER__", DATA_MARKER)


def build_html(groups: list[dict]) -> str:
    model_labels = {mid: meta["label"] for mid, meta in MODEL_META.items()}
    model_colors = {mid: meta["color"] for mid, meta in MODEL_META.items()}
    bl_labels = {k: meta["label"] for k, meta in BASELINE_META.items()}
    bl_colors = {k: meta["color"] for k, meta in BASELINE_META.items()}

    # Only emit metadata for models that actually appear in the data
    present_models = {m["model_id"] for g in groups for m in g["models"]}
    model_labels = {k: v for k, v in model_labels.items() if k in present_models}
    model_colors = {k: v for k, v in model_colors.items() if k in present_models}

    data_js = "\n".join(
        [
            f"const GROUPS = {json.dumps(groups, ensure_ascii=False)};",
            f"const MODEL_LABELS = {json.dumps(model_labels, indent=2, ensure_ascii=False)};",
            f"const MODEL_COLORS = {json.dumps(model_colors, indent=2, ensure_ascii=False)};",
            f"const BL_LABELS = {json.dumps(bl_labels, ensure_ascii=False)};",
            f"const BL_COLORS = {json.dumps(bl_colors, ensure_ascii=False)};",
            "",
            "let activeModels   = new Set(Object.keys(MODEL_LABELS));",
            "let activeDatasets = new Set(GROUPS.map(g => g.dataset));",
        ]
    )

    hljs_base = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>skribe — Prompt Inspector</title>
        <link rel="stylesheet" href="{hljs_base}/styles/github.min.css">
        <script src="{hljs_base}/highlight.min.js"></script>
        <script src="{hljs_base}/languages/python.min.js"></script>
        <style>
        {CSS}
        </style>
        </head>
        <body>

        <header>
          <div>
            <h1>skribe — Prompt Inspector</h1>
            <p>Prompts &amp; generated heuristics grouped by dataset, with baseline reference scores</p>
          </div>
        </header>

        <div class="filter-bar">
          <span class="filter-label">Models</span>
          <div class="chips" id="model-filters"></div>
          <span class="filter-label" style="margin-left:8px">Datasets</span>
          <div class="chips" id="dataset-filters"></div>
        </div>

        <main id="main"></main>

        <script>
        {data_js}

        {JS}
        </script>
        </body>
        </html>
        """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache-dir",
        default="artifacts/benchmark_results/cache",
        help="directory containing per-run JSON cache files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/skribe_inspector.html",
        help="output HTML file (default: %(default)s)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f"cache-dir not found: {cache_dir}")

    print(f"Loading cache from {cache_dir} …")
    baselines, results = load_cache(cache_dir)
    groups = build_groups(baselines, results)

    n_models = len({m["model_id"] for g in groups for m in g["models"]})
    n_entries = sum(len(g["models"]) for g in groups)
    print(f"  {len(groups)} datasets · {n_models} models · {n_entries} entries")

    html = build_html(groups)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Written → {out}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
