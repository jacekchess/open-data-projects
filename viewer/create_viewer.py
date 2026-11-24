#!/usr/bin/env python3
"""
Build a self-contained HTML player-ranking viewer.

Default inputs:
    open-data-projects/data/master/master_players.csv
    open-data-projects/master_data/position_roles_config.yaml

Default output:
    open-data-projects/data/master/player_viewer.html

Run from the repository root:
    python build_player_viewer.py

Optional overrides:
    python build_player_viewer.py --data path/to/master.csv
    python build_player_viewer.py --config path/to/config.yaml
    python build_player_viewer.py --out path/to/player_viewer.html
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("open-data-projects")

MASTER_DIR = PROJECT_DIR / "data" / "master"
MASTER_DATA_FILE = MASTER_DIR / "master_players.csv"

CONFIG_FILE = PROJECT_DIR / "master_data" / "position_roles_config.yaml"

HTML_OUTPUT = PROJECT_DIR / "viewer" / "player_viewer.html"


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

TIERS = ("high", "low", "support")

TIER_CODE = {
    "high": "h",
    "low": "l",
    "support": "s",
}

TIER_RANK = {
    "high": 0,
    "low": 1,
    "support": 2,
}

DEFAULT_WEIGHTS = {
    "high": 2.0,
    "low": 1.0,
    "support": 0.0,
}

# Position-group layout.
# Groups with left/right roles are shown on both sides of the pitch.
GROUP_LAYOUT = {
    "Goalkeeper": [
        (50, 91, "GK"),
    ],
    "Centre Back": [
        (32, 79, "LCB"),
        (68, 79, "RCB"),
    ],
    "Full Back": [
        (10, 68, "LB"),
        (90, 68, "RB"),
    ],
    "Wing Back": [
        (7, 54, "LWB"),
        (93, 54, "RWB"),
    ],
    "Defensive Midfield": [
        (50, 65, "DM"),
    ],
    "Central Midfield": [
        (32, 49, "LCM"),
        (68, 49, "RCM"),
    ],
    "Attacking Midfield": [
        (50, 35, "AM"),
    ],
    "Wide Midfield": [
        (10, 42, "LM"),
        (90, 42, "RM"),
    ],
    "Winger": [
        (13, 22, "LW"),
        (87, 22, "RW"),
    ],
    "Centre Forward": [
        (50, 8, "CF"),
    ],
}
GROUP_COLORS = [
    "#b45309", "#0369a1", "#0f766e", "#7c3aed", "#be123c",
    "#4d7c0f", "#c2410c", "#1d4ed8", "#a21caf", "#475569",
]


# --------------------------------------------------------------------------- loading
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".parquet":
        return pd.read_parquet(p)
    with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
    sep = max(["\t", ",", ";", "|"], key=first.count)
    return pd.read_csv(p, sep=sep, encoding="utf-8-sig", low_memory=False)


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(cfg, dict):
        sys.exit("Config error: YAML must contain a dictionary.")

    for key in ("position_groups", "roles"):
        if key not in cfg:
            sys.exit(f"Config error: missing '{key}' section")

    # Importance weights
    weights = dict(DEFAULT_WEIGHTS)

    for tier, spec in (cfg.get("importance_tiers") or {}).items():
        if tier not in TIERS:
            continue

        spec = spec or {}
        weight = float(spec.get("weight", DEFAULT_WEIGHTS[tier]))

        weights[tier] = (
            weight if spec.get("scored", weight > 0) else 0.0
        )

    cfg["_weights"] = weights
    cfg["_lower"] = set(cfg.get("lower_is_better") or [])

    # Convert every role into the internal high/low/support structure.
    for role, spec in cfg["roles"].items():
        if not isinstance(spec, dict):
            sys.exit(f"Config error: role '{role}' must be a dictionary.")

        metrics = spec.get("metrics")

        if isinstance(metrics, dict):
            # Already in high/low/support format.
            role_metrics = {
                tier: list(metrics.get(tier) or [])
                for tier in TIERS
            }

        elif isinstance(metrics, list):
            # Earlier project format:
            # metrics: [...]
            # metric_importance:
            #   metric_name: high / low / support

            importance = spec.get("metric_importance") or {}

            role_metrics = {
                tier: [] for tier in TIERS
            }

            for metric in metrics:
                tier = importance.get(metric, "high")

                if tier not in TIERS:
                    sys.exit(
                        f"Config error: invalid importance '{tier}' "
                        f"for metric '{metric}' in role '{role}'."
                    )

                role_metrics[tier].append(metric)

        else:
            sys.exit(
                f"Config error: role '{role}' needs a metrics list "
                "or metrics dictionary."
            )

        spec["_m"] = role_metrics

    return cfg

# --------------------------------------------------------------------------- helpers
def num_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def clean(x, nd: int = 2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def display_name(row_nick, row_name) -> str:
    nick = "" if pd.isna(row_nick) else str(row_nick).strip()
    if nick and nick.lower() != "nan":
        return nick
    return "" if pd.isna(row_name) else str(row_name).strip()


def detect_percentile_scale(df: pd.DataFrame, cfg: dict) -> float:
    cols = set()
    for spec in cfg["roles"].values():
        for t in ("high", "low"):
            cols.update(f"{m}_percentile" for m in spec["_m"][t])
    mx = 0.0
    for c in cols & set(df.columns):
        v = num_array(df, c)
        if np.isfinite(v).any():
            mx = max(mx, np.nanmax(v))
    return 100.0 if 0 < mx <= 1.0001 else 1.0


# --------------------------------------------------------------------------- scoring
def score_roles(sub, roles, cfg, scale, invert, min_cov, warnings):
    """Return array (n_players, n_roles) of integer scores 1-100 (NaN where not scorable)."""
    n = len(sub)
    out = np.full((n, len(roles)), np.nan)
    weights, lower = cfg["_weights"], cfg["_lower"]
    for j, role in enumerate(roles):
        num, den, total = np.zeros(n), np.zeros(n), 0.0
        for tier in ("high", "low"):
            w = weights[tier]
            if w <= 0:
                continue
            for m in cfg["roles"][role]["_m"][tier]:
                col = f"{m}_percentile"
                if col not in sub.columns:
                    warnings.add(f"no column '{col}' - metric '{m}' ignored in role '{role}'")
                    continue
                total += w
                v = np.clip(num_array(sub, col) * scale, 0, 100)
                if invert and m in lower:
                    v = 100 - v
                ok = np.isfinite(v)
                num[ok] += w * v[ok]
                den[ok] += w
        if total <= 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(den / total >= min_cov, num / den, np.nan)
        out[:, j] = np.clip(np.rint(1 + 99 * mean / 100), 1, 100)
    return out


def ordered_metrics(roles, cfg):
    """Union of all metrics of the roles, high-first, then by how many roles use them."""
    info = {}
    for order, role in enumerate(roles):
        for tier in TIERS:
            for m in cfg["roles"][role]["_m"][tier]:
                d = info.setdefault(m, {"rank": 9, "count": 0, "first": len(info)})
                d["rank"] = min(d["rank"], TIER_RANK[tier])
                d["count"] += 1
    return sorted(info, key=lambda m: (info[m]["rank"], -info[m]["count"], info[m]["first"]))


# --------------------------------------------------------------------------- build payload
def build_payload(df, cfg, args):
    warnings: set[str] = set()
    for col in ("position", "player_name"):
        if col not in df.columns:
            sys.exit(f"Data error: column '{col}' not found in the master table")
    df = df.copy()
    df["position"] = df["position"].astype(str).str.strip()

    scale = detect_percentile_scale(df, cfg)
    groups_cfg = cfg["position_groups"]
    pos_to_group = {}
    for g, spec in groups_cfg.items():
        for p in spec["positions"]:
            pos_to_group.setdefault(p, g)

    all_positions = list(dict.fromkeys(
        [p for spec in groups_cfg.values() for p in spec["positions"]] + list(cfg.get("source_positions") or [])
    ))
    for p in sorted(set(df["position"]) - set(pos_to_group) - {"nan"}):
        warnings.add(f"position '{p}' in the data is not mapped to any group in the YAML (ignored)")

    group_colors = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups_cfg)}
    groups_out, players_out = {}, {}

    for g, spec in groups_cfg.items():
        roles = [r for r in spec["roles"] if r in cfg["roles"]]
        for r in spec["roles"]:
            if r not in cfg["roles"]:
                warnings.add(f"role '{r}' of group '{g}' has no metric definition (skipped)")
        positions = [p for p in spec["positions"]]
        sub = df[df["position"].isin(positions)].reset_index(drop=True)

        metrics = []
        for m in ordered_metrics(roles, cfg):
            if any(c in df.columns for c in (m, f"{m}_per90", f"{m}_percentile")):
                metrics.append(m)
            else:
                warnings.add(f"metric '{m}' has no column in the data (dropped from '{g}')")
        midx = {m: i for i, m in enumerate(metrics)}
        role_tiers = {}
        for r in roles:
            codes = [""] * len(metrics)
            for tier in TIERS:  # high -> low -> support; first assignment wins
                for m in cfg["roles"][r]["_m"][tier]:
                    if m in midx and not codes[midx[m]]:
                        codes[midx[m]] = TIER_CODE[tier]
            role_tiers[r] = codes
        groups_out[g] = {"roles": roles, "metrics": metrics, "roleTiers": role_tiers,
                         "color": group_colors[g]}

        if sub.empty or not roles:
            for p in positions:
                players_out[p] = []
            continue

        scores = score_roles(sub, roles, cfg, scale, not args.percentiles_pre_inverted,
                             args.min_coverage, warnings)
        has_any = np.isfinite(scores).any(axis=1)
        best = np.where(has_any, np.nanargmax(np.where(np.isfinite(scores), scores, -1), axis=1), -1)

        cols = {}
        for m in metrics:
            pct = num_array(sub, f"{m}_percentile") * scale if f"{m}_percentile" in sub.columns else np.full(len(sub), np.nan)
            cols[m] = (pct, num_array(sub, f"{m}_per90"), num_array(sub, m))

        minutes = num_array(sub, "total_minutes")
        matches = num_array(sub, "matches_played")
        nick = sub["nick_name"] if "nick_name" in sub.columns else pd.Series([np.nan] * len(sub))
        pname = sub["player_name"]
        team = sub["most_played_team"] if "most_played_team" in sub.columns else pd.Series([""] * len(sub))

        for p in positions:
            rows = []
            for i in np.flatnonzero((sub["position"] == p).to_numpy()):
                rows.append({
                    "n": display_name(nick.iloc[i], pname.iloc[i]),
                    "p": p,
                    "t": "" if pd.isna(team.iloc[i]) else str(team.iloc[i]),
                    "min": int(minutes[i]) if np.isfinite(minutes[i]) else 0,
                    "mp": int(matches[i]) if np.isfinite(matches[i]) else None,
                    "r": roles[best[i]] if best[i] >= 0 else None,
                    "s": int(scores[i, best[i]]) if best[i] >= 0 else None,
                    "rs": [int(x) if np.isfinite(x) else None for x in scores[i]],
                    "v": [[clean(cols[m][0][i], 1), clean(cols[m][1][i]), clean(cols[m][2][i])] for m in metrics],
                })
            players_out[p] = rows

        # Build pitch positions using YAML position groups, not individual
    # StatsBomb positions.
    positions_out = []

    for group, slots in GROUP_LAYOUT.items():
        if group not in groups_cfg:
            continue

        group_positions = groups_cfg[group]["positions"]

        # Number of player rows across all underlying positions.
        player_count = sum(
            len(players_out.get(position, []))
            for position in group_positions
        )

        for x, y, abbreviation in slots:
            positions_out.append({
                "name": group,
                "abbr": abbreviation,
                "x": x,
                "y": y,
                "group": group,
                "color": group_colors[group],
                "count": player_count,
                "positions": group_positions,
            })

    # Warn if a YAML group does not have a pitch layout.
    for group in groups_cfg:
        if group not in GROUP_LAYOUT:
            warnings.add(
                f"no pitch coordinates for position group '{group}' "
                "(add it to GROUP_LAYOUT)"
            )

    payload = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "weights": cfg["_weights"], "minCoverage": args.min_coverage,
            "minMinutes": args.min_minutes, "rows": int(len(df)),
        },
        "positions": positions_out,
        "groups": groups_out,
        "players": players_out,
    }
    return payload, warnings


# --------------------------------------------------------------------------- HTML
TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Player ranking viewer</title>
<style>
:root{
  --bg:#f3f4f6; --card:#fff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --head:#f9fafb;
  --high:rgba(34,197,94,.42); --low:rgba(59,130,246,.28); --support:rgba(168,85,247,.28);
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0f1115; --card:#181b21; --ink:#e5e7eb; --muted:#9ca3af; --line:#2a2f38; --head:#1f232b;
    --high:rgba(34,197,94,.45); --low:rgba(59,130,246,.38); --support:rgba(168,85,247,.38); }
}
*{box-sizing:border-box}
body{margin:0;font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header.top{display:flex;align-items:center;gap:24px;padding:12px 24px;background:var(--card);border-bottom:1px solid var(--line)}
header.top h1{font-size:17px;margin:0}
nav button{font:inherit;border:0;background:none;padding:6px 12px;border-bottom:2px solid transparent;color:var(--muted);cursor:pointer}
nav button.active{color:var(--ink);border-color:#16a34a;font-weight:600}
main{max-width:1500px;margin:0 auto;padding:20px 24px 60px}
.top-row{display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap}
.pitch{position:relative;width:min(560px,100%);aspect-ratio:100/117;border-radius:10px;overflow:hidden;
  background:repeating-linear-gradient(180deg,#2f7d3b 0 8.33%,#2a7236 8.33% 16.66%);box-shadow:0 2px 10px rgba(0,0,0,.25)}
.pitch svg{position:absolute;inset:0;width:100%;height:100%}
.dot{position:absolute;transform:translate(-50%,-50%);width:40px;height:40px;border-radius:50%;border:2px solid #fff;
  background:var(--c);color:#fff;font:700 11px system-ui;cursor:pointer;display:flex;align-items:center;justify-content:center;
  box-shadow:0 1px 4px rgba(0,0,0,.45);transition:transform .12s, box-shadow .12s;padding:0}
.dot:hover:not(:disabled){transform:translate(-50%,-50%) scale(1.15)}
.dot.sel{box-shadow:0 0 0 4px rgba(250,204,21,.95),0 1px 4px rgba(0,0,0,.45);transform:translate(-50%,-50%) scale(1.15)}
.dot.empty{opacity:.35;cursor:default}
.dot i{position:absolute;top:-7px;right:-9px;font:600 10px system-ui;background:#111827;color:#fff;border-radius:9px;padding:1px 5px;font-style:normal}
.info{flex:1;min-width:280px;max-width:520px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.info h2{font-size:15px;margin:0 0 8px}
.info p{margin:6px 0;color:var(--muted)}
.legend{display:flex;flex-direction:column;gap:6px;margin:10px 0}
.chip{display:inline-block;min-width:46px;text-align:center;padding:2px 8px;border-radius:4px;margin-right:8px;font-weight:600}
.t-h,.chip.h,th.h-h{background:var(--high)} .t-l,.chip.l,th.h-l{background:var(--low)} .t-s,.chip.s,th.h-s{background:var(--support)}
.glegend{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
.glegend span{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}
.glegend b{width:12px;height:12px;border-radius:50%;display:inline-block}
#results{margin-top:26px}
.toolbar{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin-bottom:10px}
.toolbar h2{margin:0;font-size:18px;flex:1 1 100%}
.toolbar .sub{color:var(--muted);font-size:12px;flex:1 1 100%;margin-top:-10px}
.toolbar label{display:flex;flex-direction:column;font-size:11px;color:var(--muted);gap:3px}
.toolbar input,.toolbar select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.toolbar input[type=number]{width:90px}
.table-wrap{overflow:auto;max-height:75vh;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}
th{position:sticky;top:0;z-index:2;background:var(--head);font-size:12px;cursor:pointer;user-select:none;vertical-align:bottom}
th.metric{white-space:normal;min-width:74px;max-width:96px;text-align:center}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.fb{font-style:italic;opacity:.75}
.c0{position:sticky;left:0;z-index:1;background:var(--card);min-width:40px}
.c1{position:sticky;left:40px;z-index:1;background:var(--card);font-weight:600;border-right:1px solid var(--line)}
th.c0,th.c1{z-index:3;background:var(--head)}
tbody tr:hover td{filter:brightness(.96)}
.score{display:inline-block;min-width:34px;text-align:center;padding:2px 6px;border-radius:5px;color:#fff;font-weight:700}
.empty-msg{padding:30px;text-align:center;color:var(--muted)}
</style>
</head>
<body>
<header class="top">
  <h1>Player viewer</h1>
  <nav><button class="active" data-page="ranking">Player ranking</button></nav>
</header>
<main>
<section id="page-ranking">
  <div class="top-row">
    <div class="pitch" id="pitch">
      <svg viewBox="0 0 100 117" preserveAspectRatio="none" fill="none" stroke="rgba(255,255,255,.75)" stroke-width=".4">
        <rect x="2" y="2" width="96" height="113"/>
        <line x1="2" y1="58.5" x2="98" y2="58.5"/>
        <circle cx="50" cy="58.5" r="10"/><circle cx="50" cy="58.5" r=".6" fill="#fff"/>
        <rect x="22" y="2" width="56" height="18"/><rect x="37" y="2" width="26" height="6"/>
        <rect x="22" y="97" width="56" height="18"/><rect x="37" y="109" width="26" height="6"/>
        <circle cx="50" cy="14" r=".6" fill="#fff"/><circle cx="50" cy="103" r=".6" fill="#fff"/>
        <path d="M41 20 A10 10 0 0 0 59 20"/><path d="M41 97 A10 10 0 0 1 59 97"/>
      </svg>
    </div>
    <aside class="info">
      <h2>Player ranking</h2>
      <p>Click a position on the pitch to list the players who played there, ranked by the score of their best-fitting role.</p>
      <div class="legend">
        <div><span class="chip h">High</span>important for the player's best role &ndash; scored</div>
        <div><span class="chip l">Low</span>nice to have in that role &ndash; scored</div>
        <div><span class="chip s">Support</span>role feature, shown only &ndash; not scored</div>
        <div><span class="chip" style="border:1px solid var(--line)">&nbsp;</span>metric not used by the player's best role</div>
      </div>
      <p id="scoring"></p>
      <div class="glegend" id="glegend"></div>
    </aside>
  </div>

  <section id="results" hidden>
    <div class="toolbar">
      <h2 id="res-title"></h2>
      <div class="sub" id="res-sub"></div>
      <label>Search<input id="q" type="search" placeholder="player name"></label>
      <label>Position<select id="positionSel"><option value="">All positions</option></select></label>
      <label>Best role<select id="roleSel"></select></label>
      <label>Min. minutes<input id="minMin" type="number" min="0" step="90"></label>
      <label>Values<select id="modeSel">
        <option value="pct">Percentile</option><option value="p90">Per 90</option><option value="raw">Season total</option></select></label>
      <span id="count" style="color:var(--muted);font-size:12px;padding-bottom:7px"></span>
    </div>
    <div class="table-wrap"><table id="tbl"></table></div>
  </section>
</section>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {pos:null, sourcePos:'', sortKey:'s', dir:-1, mode:'pct', role:'', q:'', minMin:D.meta.minMinutes || 0};
const WORDS = {np:'NP', xag:'xAG', xg:'xG', pct:'%', gk:'GK'};
const TIER_NAME = {h:'high', l:'low', s:'support'};

function pretty(m){
  const t = m.split('_').map(w => WORDS[w] || w).join(' ');
  return t.charAt(0).toUpperCase() + t.slice(1);
}
function fmt(n, isPct){
  if (n == null) return '–';
  if (isPct) return String(Math.round(n));
  if (Math.abs(n) >= 100 || Number.isInteger(n)) return String(Math.round(n));
  return n.toFixed(2);
}
function disp(v){            // v = [percentile, per90, raw]
  let n, fb = false;
  if (state.mode === 'pct'){ if (v[0] != null) n = v[0]; else { n = v[1] != null ? v[1] : v[2]; fb = true; } }
  else if (state.mode === 'p90'){ n = v[1] != null ? v[1] : v[2]; }
  else n = v[2];
  return [fmt(n, state.mode === 'pct' && !fb), n, fb];
}

/* ---- pitch ---- */
const w = D.meta.weights;
$('#scoring').textContent =
  `Score = weighted mean of percentile metrics (high ×${w.high}, low ×${w.low}) of a role, mapped to an integer 1–100. ` +
  `A role needs at least ${Math.round(D.meta.minCoverage*100)}% of its metric weight available. Generated ${D.meta.generated}.`;
const pitch = $('#pitch');
D.positions.forEach(p => {
  const b = document.createElement('button');
  b.className = 'dot' + (p.count ? '' : ' empty');
  b.style.left = p.x + '%'; b.style.top = p.y + '%'; b.style.setProperty('--c', p.color);
  b.innerHTML = `<span>${esc(p.abbr)}</span>` + (p.count ? `<i>${p.count}</i>` : '');
  b.title = `${p.name} – ${p.count} player rows`;
  b.dataset.pos = p.name; b.disabled = !p.count;
  b.onclick = () => selectPos(p.name);
  pitch.appendChild(b);
});
$('#glegend').innerHTML = Object.entries(D.groups).map(([g, s]) =>
  `<span><b style="background:${s.color}"></b>${esc(g)}</span>`).join('');

/* ---- selection ---- */
function pPositionsForGroup(name){
  const pitchPosition = D.positions.find(x => x.name === name);
  return pitchPosition ? pitchPosition.positions : [];
}
function selectPos(name){
  state.pos = name;
  state.sortKey = 's';
  state.dir = -1;
  state.role = '';
  state.q = '';
  state.sourcePos = '';

  $('#q').value = '';
  $('#positionSel').innerHTML =
    '<option value="">All positions</option>' +
    pPositionsForGroup(name).map(pos => `<option value="${esc(pos)}">${esc(pos)}</option>`).join('');

  document.querySelectorAll('.dot').forEach(d =>
    d.classList.toggle('sel', d.dataset.pos === name)
  );

  const p = D.positions.find(x => x.name === name);
  const g = D.groups[p.group];

  $('#roleSel').innerHTML =
    '<option value="">All roles</option>' +
    g.roles.map(r => `<option>${esc(r)}</option>`).join('');

  $('#res-title').textContent = name;
  $('#res-sub').textContent =
    'Roles: ' + g.roles.join(' · ');

  $('#minMin').value = state.minMin;
  $('#results').hidden = false;

  render();

  $('#results').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}

/* ---- table ---- */
function scoreBadge(r){
  if (r.s == null) return '–';
  const tip = D.groups[D.positions.find(x => x.name === state.pos).group].roles
    .map((ro, k) => `${ro}: ${r.rs[k] == null ? '–' : r.rs[k]}`).join('\n');
  return `<span class="score" style="background:hsl(${Math.round(r.s*1.2)} 62% 40%)" title="${esc(tip)}">${r.s}</span>`;
}
function render(){
  const p = D.positions.find(x => x.name === state.pos);
  const g = D.groups[p.group];

  // Combine all individual StatsBomb positions belonging to this group.
  const all = p.positions.flatMap(position =>
    D.players[position] || []
  );
  const rows = all.filter(r =>
    r.min >= state.minMin &&
    (!state.role || r.r === state.role) &&
    (!state.sourcePos || r.p === state.sourcePos) &&
    (!state.q || r.n.toLowerCase().includes(state.q))
  );
  const key = state.sortKey;
  const get = r => key === 's' ? r.s : key === 'n' ? r.n.toLowerCase() : key === 't' ? (r.t || '').toLowerCase()
                 : key === 'r' ? r.r : key === 'min' ? r.min : disp(r.v[+key.slice(1)])[1];
  rows.sort((a, b) => {
    const x = get(a), y = get(b);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * state.dir;
  });

  const th = (k, label, cls, title) =>
    `<th class="${cls || ''}" data-k="${k}" title="${esc(title || label)}">${esc(label)}${state.sortKey === k ? (state.dir < 0 ? ' ▼' : ' ▲') : ''}</th>`;
  let h = '<thead><tr>' + th('#', '#', 'c0') + th('n', 'Player', 'c1') + th('t', 'Team') + th('min', 'Min.') +
          th('r', 'Best role') + th('s', 'Score');
  g.metrics.forEach((m, i) => {
    const t = state.role ? g.roleTiers[state.role][i] : '';
    h += th('m' + i, pretty(m), 'metric' + (t ? ' h-' + t : ''), m + (t ? ` (${TIER_NAME[t]} for ${state.role})` : ''));
  });
  h += '</tr></thead><tbody>';
  if (!rows.length) h += `<tr><td colspan="${6 + g.metrics.length}" class="empty-msg">No players match the filters.</td></tr>`;
  rows.forEach((r, ri) => {
    const tiers = r.r ? g.roleTiers[r.r] : null;
    let tr = `<tr><td class="num c0">${ri + 1}</td><td class="c1">${esc(r.n)}</td><td>${esc(r.t)}</td>` +
             `<td class="num">${r.min}</td><td>${esc(r.r || '–')}</td><td class="num">${scoreBadge(r)}</td>`;
    g.metrics.forEach((m, i) => {
      const [txt, n, fb] = disp(r.v[i]);
      const t = tiers ? tiers[i] : '';
      const tip = `${pretty(m)}${t ? ' – ' + TIER_NAME[t] + ' for ' + r.r : ''}\n` +
                  `percentile ${fmt(r.v[i][0], true)} · per 90 ${fmt(r.v[i][1])} · total ${fmt(r.v[i][2])}`;
      tr += `<td class="num${t ? ' t-' + t : ''}${fb ? ' fb' : ''}" title="${esc(tip)}">${txt}</td>`;
    });
    h += tr + '</tr>';
  });
  $('#tbl').innerHTML = h + '</tbody>';
  $('#count').textContent = `${rows.length} of ${all.length} players`;
  $('#tbl thead').onclick = e => {
    const c = e.target.closest('th'); if (!c || c.dataset.k === '#') return;
    const k = c.dataset.k;
    if (state.sortKey === k) state.dir = -state.dir;
    else { state.sortKey = k; state.dir = (k === 'n' || k === 't' || k === 'r') ? 1 : -1; }
    render();
  };
}

/* ---- filter controls ---- */
$('#q').addEventListener('input', e => {
  state.q = e.target.value.trim().toLowerCase();
  if (state.pos) render();
});
$('#positionSel').addEventListener('change', e => {
  state.sourcePos = e.target.value;
  if (state.pos) render();
});
$('#roleSel').addEventListener('change', e => {
  state.role = e.target.value;
  if (state.pos) render();
});
$('#minMin').addEventListener('input', e => {
  const value = Number(e.target.value);
  state.minMin = e.target.value === '' || !Number.isFinite(value) ? 0 : Math.max(0, value);
  if (state.pos) render();
});
$('#modeSel').addEventListener('change', e => {
  state.mode = e.target.value;
  if (state.pos) render();
});
})();
</script>
</body>
</html>
'''


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the HTML player ranking viewer."
    )

    ap.add_argument(
        "--data",
        type=Path,
        default=MASTER_DATA_FILE,
        help=f"Master player table (default: {MASTER_DATA_FILE})",
    )

    ap.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help=f"Position and role configuration (default: {CONFIG_FILE})",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=HTML_OUTPUT,
        help=f"HTML output file (default: {HTML_OUTPUT})",
    )

    ap.add_argument(
        "--min-minutes",
        type=int,
        default=0,
        help="Default minimum minutes filter in the viewer.",
    )

    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.6,
        help=(
            "Share of a role's metric weight that must be available "
            "to score it (default: 0.6)."
        ),
    )

    ap.add_argument(
        "--percentiles-pre-inverted",
        action="store_true",
        help=(
            "Set if lower-is-better metric percentiles are already "
            "inverted in the master table."
        ),
    )

    args = ap.parse_args()

    # Validate input files
    if not args.data.exists():
        sys.exit(f"Master player file not found: {args.data}")

    if not args.config.exists():
        sys.exit(f"Position-role config not found: {args.config}")

    if not 0 <= args.min_coverage <= 1:
        sys.exit("--min-coverage must be between 0 and 1.")

    # Load data and configuration
    print(f"Loading master player table: {args.data}")
    df = load_table(str(args.data))

    print(f"Loading position-role configuration: {args.config}")
    cfg = load_config(args.config)

    print("Building player ranking data...")
    payload, warnings = build_payload(df, cfg, args)

    # Create HTML output directory if necessary
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Embed data in the HTML
    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    # Prevent embedded data from accidentally closing the script tag
    data = data.replace("</", "<\\/")

    html = TEMPLATE.replace("__DATA__", data)

    args.out.write_text(html, encoding="utf-8")

    n_rows = sum(
        len(players)
        for players in payload["players"].values()
    )

    print()
    print("Player viewer created successfully.")
    print(f"Output: {args.out}")
    print(f"Player-position rows: {n_rows}")
    print(f"Position groups: {len(payload['groups'])}")
    print(f"File size: {args.out.stat().st_size / 1e6:.2f} MB")

    if warnings:
        print()
        print("Warnings:")

        for warning in sorted(warnings):
            print(f"  - {warning}")

if __name__ == "__main__":
    main()