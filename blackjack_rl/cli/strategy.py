"""Print an agent's playing strategy as the classic hard/soft/pairs tables and its bet spread,
compare it against basic strategy, and optionally write a self-contained HTML report."""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import os
import webbrowser
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..agents.basic_strategy import basic_strategy, hi_lo_bet_index
from ..engine import ACTION_LETTERS, ACTION_NAMES, Action, Rules
from ..env.blackjack_env import N_PLAY_ACTIONS
from ..env.observation import encode_observation
from .common import add_rules_args, env_from_args, make_agent

DEALER_UPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # 1 = Ace, shown last
DEALER_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]
TC_RANGE = list(range(-5, 9))
LETTER_NAMES = {"S": "stand", "H": "hit", "D": "double", "P": "split", "R": "surrender"}


# ---------------------------------------------------------------------------- data model
@dataclass
class Cell:
    action: str                       # agent's letter
    basic: str                        # basic strategy letter
    q: Optional[Dict[str, float]] = None  # RL scores per legal action name (Q in bet units, or policy probability)

    @property
    def agrees(self) -> bool:
        return self.action == self.basic


@dataclass
class StrategyReport:
    agent_name: str
    checkpoint: Optional[str]
    rules: Rules
    bet_sizes: Tuple[float, ...]
    true_count: float
    decks_frac: float
    hard: Dict[Tuple[int, int], Cell] = field(default_factory=dict)    # (total, dealer) -> Cell
    soft: Dict[Tuple[int, int], Cell] = field(default_factory=dict)    # (total, dealer)
    pairs: Dict[Tuple[int, int], Cell] = field(default_factory=dict)   # (pair value, dealer)
    agent_bets: Optional[List[float]] = None                            # per TC in TC_RANGE (None for basic)
    hilo_bets: List[float] = field(default_factory=list)
    bet_q: Optional[List[List[float]]] = None                           # [tc][bet index] (RL agents only)
    compare: bool = True
    score_kind: str = "q"                                               # "q" (DQN) or "prob" (PPO)

    def agreement(self, table: Dict) -> Tuple[int, int]:
        cells = list(table.values())
        return sum(c.agrees for c in cells), len(cells)

    @property
    def hard_rows(self):
        return [(str(t), t) for t in range(5, 20)]

    @property
    def soft_rows(self):
        return [(f"A,{t - 11}", t) for t in range(13, 21)]

    @property
    def pair_rows(self):
        return [(("A,A" if p == 1 else f"{'T' if p == 10 else p},{'T' if p == 10 else p}"), p)
                for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]]


def _legal_for(rules: Rules, is_pair: bool) -> List[Action]:
    legal = [Action.STAND, Action.HIT, Action.DOUBLE]
    if is_pair:
        legal.append(Action.SPLIT)
    if rules.surrender:
        legal.append(Action.SURRENDER)
    return legal


def _mask(legal: List[Action], n_actions: int) -> np.ndarray:
    m = np.zeros(n_actions, dtype=np.int8)
    m[[int(a) for a in legal]] = 1
    return m


def build_report(agent_kind: str, agent, env, true_count: float, decks_frac: float,
                 checkpoint: Optional[str], compare: bool = True) -> StrategyReport:
    rules = env.rules
    n_actions = env.action_space.n
    bet_frac = env.bet_sizes[0] / env.max_bet
    is_rl = agent_kind in ("dqn", "ppo", "rl")
    if is_rl:
        agent_kind = getattr(agent, "name", agent_kind)          # resolve "rl" to the checkpoint's kind
    score_kind = getattr(agent, "score_kind", "q") if is_rl else "q"
    score_scale = env.max_bet if score_kind == "q" else 1.0

    def decide(total, is_soft, is_pair, dealer, legal) -> Tuple[Action, Optional[Dict[str, float]]]:
        if is_rl:
            obs = encode_observation(
                phase=1, player_total=total, is_soft=is_soft, is_pair=is_pair,
                can_double=Action.DOUBLE in legal, can_split=Action.SPLIT in legal,
                can_surrender=Action.SURRENDER in legal, is_split_hand=False, dealer_upcard=dealer,
                true_count=true_count, decks_frac=decks_frac, bet_frac=bet_frac,
                num_hands=1, max_splits=rules.max_splits)
            mask = _mask(legal, n_actions)
            sc = agent.action_scores(obs, mask) * score_scale
            qd = {ACTION_NAMES[a]: float(sc[int(a)]) for a in legal}
            return Action(agent.greedy_action(obs, mask)), qd
        return basic_strategy(total, is_soft, is_pair, dealer, legal, rules), None

    def cell(total, is_soft, is_pair, dealer) -> Cell:
        legal = _legal_for(rules, is_pair)
        act, q = decide(total, is_soft, is_pair, dealer, legal)
        ref = basic_strategy(total, is_soft, is_pair, dealer, legal, rules)
        return Cell(ACTION_LETTERS[act], ACTION_LETTERS[ref], q)

    rep = StrategyReport(agent_kind, checkpoint, rules, env.bet_sizes, true_count, decks_frac,
                         compare=compare and is_rl, score_kind=score_kind)
    for total in range(5, 20):
        for d in DEALER_UPS:
            rep.hard[(total, d)] = cell(total, False, False, d)
    for total in range(13, 21):
        for d in DEALER_UPS:
            rep.soft[(total, d)] = cell(total, True, False, d)
    for pair in [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]:
        total, is_soft = (12, True) if pair == 1 else (2 * pair, False)
        for d in DEALER_UPS:
            rep.pairs[(pair, d)] = cell(total, is_soft, True, d)

    rep.hilo_bets = [env.bet_sizes[hi_lo_bet_index(tc, env.bet_sizes)] for tc in TC_RANGE]
    if is_rl:
        bet_mask = np.zeros(n_actions, dtype=np.int8)
        bet_mask[N_PLAY_ACTIONS:] = 1
        rep.agent_bets, rep.bet_q = [], []
        for tc in TC_RANGE:
            obs = encode_observation(phase=0, true_count=tc, decks_frac=decks_frac, bet_frac=0.0)
            act = agent.greedy_action(obs, bet_mask)
            rep.agent_bets.append(env.bet_sizes[act - N_PLAY_ACTIONS])
            rep.bet_q.append([float(x) for x in agent.action_scores(obs, bet_mask)[N_PLAY_ACTIONS:]])
    elif agent_kind == "hilo":
        rep.agent_bets = list(rep.hilo_bets)
    else:
        rep.agent_bets = [env.bet_sizes[0]] * len(TC_RANGE)
    return rep


# ---------------------------------------------------------------------------- text output
def _fmt_table(title: str, rows, table: Dict, compare: bool) -> str:
    out = [f"{title:<12}" + " ".join(f"{d:>3}" for d in DEALER_LABELS)]
    mism = cells = 0
    for label, key in rows:
        line = f"{label:<12}"
        for d in DEALER_UPS:
            c = table[(key, d)]
            cells += 1
            if compare and not c.agrees:
                mism += 1
                line += f" {c.action}/{c.basic}"[:4].rjust(4)
            else:
                line += f"{c.action:>4}"
        out.append(line)
    if compare:
        out.append(f"{'':<12}agreement with basic strategy: {cells - mism}/{cells} "
                   f"({100 * (cells - mism) / cells:.1f}%)   (cells shown as agent/basic where they differ)")
    return "\n".join(out)


def render_text(rep: StrategyReport) -> str:
    lines = [f"Rules: {rep.rules.describe()}",
             f"Agent: {rep.agent_name}{' (' + rep.checkpoint + ')' if rep.checkpoint else ''}   "
             f"assumed true count {rep.true_count:+.1f}, {rep.decks_frac:.0%} of shoe left",
             "Legend: S=stand H=hit D=double P=split R=surrender  (2-card hands; dealer up-card across the top)", ""]
    lines.append(_fmt_table("HARD", rep.hard_rows, rep.hard, rep.compare))
    lines.append("")
    lines.append(_fmt_table("SOFT", rep.soft_rows, rep.soft, rep.compare))
    lines.append("")
    lines.append(_fmt_table("PAIRS", rep.pair_rows, rep.pairs, rep.compare))
    lines.append("\nBET by true count:")
    lines.append("TC        " + " ".join(f"{tc:>+5d}" for tc in TC_RANGE))
    label = {"hilo": "hi-lo", "basic": "flat"}.get(rep.agent_name, "agent")
    lines.append(f"{label:<10}" + " ".join(f"{b:>5g}" for b in rep.agent_bets))
    if rep.bet_q:
        lines.append("hi-lo ref " + " ".join(f"{b:>5g}" for b in rep.hilo_bets))
        what = "Q-values (scaled reward units)" if rep.score_kind == "q" else "policy probability"
        lines.append(f"\n{what} per bet size:")
        for i, b in enumerate(rep.bet_sizes):
            lines.append(f"bet {b:<6g}" + " ".join(f"{row[i]:>+5.3f}" for row in rep.bet_q))
    return "\n".join(lines)


# ---------------------------------------------------------------------------- HTML output
_CSS = """
:root { color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;   /* categorical slots */
  --crit: #d03b3b; --good: #006300; --div-neg: #e34948; --div-pos: #2a78d6; --div-mid: #f0efec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --crit: #d03b3b; --good: #0ca30c; --div-neg: #e66767; --div-pos: #3987e5; --div-mid: #383835; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 28px 22px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 10px; letter-spacing: .2px; }
.meta { color: var(--ink-2); font-size: 13px; }
.meta code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 12px 14px; }
.tile .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .6px; }
.tile .value { font-size: 26px; font-weight: 700; margin-top: 2px; }
.tile .sub { color: var(--ink-2); font-size: 12px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px 18px; margin: 14px 0; overflow-x: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; font-size: 13px; color: var(--ink-2); margin: 6px 0 12px; }
.legend .sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: -2px; margin-right: 6px; border: 1px solid var(--ring); }
.legend label { margin-left: auto; cursor: pointer; }
table.strat { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; }
table.strat th { font-weight: 600; color: var(--muted); font-size: 12px; padding: 4px 6px; }
table.strat th.row { text-align: right; color: var(--ink-2); font-weight: 600; }
table.strat td { width: 46px; height: 40px; text-align: center; border-radius: 6px; font-weight: 700; font-size: 15px; position: relative;
  border: 1px solid transparent; }
table.strat td small { display: block; font-size: 10px; font-weight: 600; color: var(--ink-2); line-height: 1; margin-top: 1px; }
td.S { background: color-mix(in srgb, var(--s1) 32%, var(--surface)); }
td.H { background: color-mix(in srgb, var(--s2) 32%, var(--surface)); }
td.D { background: color-mix(in srgb, var(--s3) 34%, var(--surface)); }
td.P { background: color-mix(in srgb, var(--s4) 38%, var(--surface)); }
td.R { background: color-mix(in srgb, var(--s5) 34%, var(--surface)); }
.sw.S { background: color-mix(in srgb, var(--s1) 32%, var(--surface)); } .sw.H { background: color-mix(in srgb, var(--s2) 32%, var(--surface)); }
.sw.D { background: color-mix(in srgb, var(--s3) 34%, var(--surface)); } .sw.P { background: color-mix(in srgb, var(--s4) 38%, var(--surface)); }
.sw.R { background: color-mix(in srgb, var(--s5) 34%, var(--surface)); }
td.diff { border-color: var(--crit); box-shadow: inset 0 0 0 1px var(--crit); }
body.only-diff td:not(.diff) { opacity: .28; }
.tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }
.agree { color: var(--ink-2); font-size: 12px; margin-top: 8px; }
.agree b { color: var(--ink); }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-2); }
svg .grid line { stroke: var(--grid); stroke-width: 1; } svg .axis { stroke: var(--axis); }
svg .bar-agent { fill: var(--s1); } svg .bar-ref { fill: var(--muted); opacity: .55; }
svg .lbl { fill: var(--ink); font-weight: 600; }
table.q { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; font-size: 12px; }
table.q th { color: var(--muted); font-weight: 600; padding: 3px 6px; }
table.q td { padding: 4px 6px; text-align: right; border-radius: 4px; min-width: 52px; }
table.q td.best { font-weight: 700; outline: 1px solid var(--ink-2); }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""

_JS = """
document.getElementById('only-diff').addEventListener('change', e => document.body.classList.toggle('only-diff', e.target.checked));
"""


def _cell_title(c: Cell) -> str:
    parts = [f"agent: {LETTER_NAMES[c.action]}"]
    if c.action != c.basic:
        parts.append(f"basic strategy: {LETTER_NAMES[c.basic]}")
    if c.q:
        parts.append("scores: " + "  ".join(f"{k} {v:+.3f}" for k, v in sorted(c.q.items(), key=lambda kv: -kv[1])))
    return html.escape(" · ".join(parts))


def _html_table(title: str, rows, table: Dict, compare: bool, agreement: Tuple[int, int]) -> str:
    out = [f'<div class="card"><h2>{title}</h2><table class="strat"><thead><tr><th></th>']
    out += [f"<th>{d}</th>" for d in DEALER_LABELS]
    out.append("</tr></thead><tbody>")
    for label, key in rows:
        out.append(f'<tr><th class="row">{label}</th>')
        for d in DEALER_UPS:
            c = table[(key, d)]
            cls = c.action + (" diff" if compare and not c.agrees else "")
            sub = f"<small>basic {c.basic}</small>" if compare and not c.agrees else ""
            out.append(f'<td class="{cls}" title="{_cell_title(c)}">{c.action}{sub}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    if compare:
        ok, n = agreement
        out.append(f'<div class="agree">agreement with basic strategy: <b>{ok}/{n}</b> ({100 * ok / n:.1f}%) — '
                   f'outlined cells differ (small text = basic strategy)</div>')
    out.append("</div>")
    return "".join(out)


def _bet_svg(rep: StrategyReport) -> str:
    W, H = 760, 230
    left, right, top, bottom = 44, 12, 18, 40
    pw, ph = W - left - right, H - top - bottom
    max_bet = max(rep.bet_sizes)
    n = len(TC_RANGE)
    gw = pw / n
    two = rep.bet_q is not None
    bw = min(22, (gw - 8) / (2 if two else 1))
    y = lambda v: top + ph * (1 - v / max_bet)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" aria-label="bet size by true count">']
    parts.append('<g class="grid">')
    for b in sorted(set(rep.bet_sizes)):
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{y(b):.1f}" y2="{y(b):.1f}"/>'
                     f'<text x="{left - 6}" y="{y(b) + 4:.1f}" text-anchor="end">{b:g}</text>')
    parts.append("</g>")
    parts.append(f'<line class="axis" x1="{left}" x2="{W - right}" y1="{top + ph}" y2="{top + ph}"/>')
    for i, tc in enumerate(TC_RANGE):
        cx = left + gw * (i + 0.5)
        series = [("bar-agent", rep.agent_bets[i], rep.agent_name if two else {"hilo": "Hi-Lo", "basic": "flat"}[rep.agent_name])]
        if two:
            series.append(("bar-ref", rep.hilo_bets[i], "Hi-Lo reference"))
        x0 = cx - (bw * len(series) + 2 * (len(series) - 1)) / 2
        for j, (cls, v, name) in enumerate(series):
            x = x0 + j * (bw + 2)
            h = top + ph - y(v)
            r = min(4, h)
            parts.append(f'<path class="{cls}" d="M{x:.1f},{top + ph} v{-(h - r):.1f} q0,-{r} {r},-{r} h{bw - 2 * r:.1f} '
                         f'q{r},0 {r},{r} v{h - r:.1f} z"><title>TC {tc:+d}: {name} bets {v:g}</title></path>')
            if cls == "bar-agent":
                parts.append(f'<text class="lbl" x="{x + bw / 2:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle">{v:g}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{H - 22}" text-anchor="middle">{tc:+d}</text>')
    parts.append(f'<text x="{left + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">true count</text>')
    parts.append("</svg>")
    if two:
        parts.append('<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>DQN agent</span>'
                     '<span><i class="sw" style="background:var(--muted);opacity:.55"></i>Hi-Lo 1-2-4-8 reference</span></div>')
    return "".join(parts)


def _q_table(rep: StrategyReport) -> str:
    if not rep.bet_q:
        return ""
    flat = [abs(v) for row in rep.bet_q for v in row]
    scale = max(max(flat), 1e-6)
    out = ['<table class="q"><thead><tr><th>bet \\ TC</th>'] + [f"<th>{tc:+d}</th>" for tc in TC_RANGE] + ["</tr></thead><tbody>"]
    for i, b in enumerate(rep.bet_sizes):
        out.append(f"<tr><th>{b:g}</th>")
        for t, row in enumerate(rep.bet_q):
            v = row[i]
            best = i == int(np.argmax(row))
            pct = int(min(1.0, abs(v) / scale) * 60)
            col = "var(--div-pos)" if v >= 0 else "var(--div-neg)"
            what = "Q = " if rep.score_kind == "q" else "p = "
            out.append(f'<td class="{"best" if best else ""}" style="background:color-mix(in srgb,{col} {pct}%,var(--div-mid))" '
                       f'title="TC {TC_RANGE[t]:+d}, bet {b:g}: {what}{v:+.4f}">{v:+.3f}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_html(rep: StrategyReport) -> str:
    title = f"Blackjack strategy — {rep.agent_name}"
    hard_a, soft_a, pair_a = rep.agreement(rep.hard), rep.agreement(rep.soft), rep.agreement(rep.pairs)
    tot_ok = hard_a[0] + soft_a[0] + pair_a[0]
    tot_n = hard_a[1] + soft_a[1] + pair_a[1]
    src = f" &middot; <code>{html.escape(rep.checkpoint)}</code>" if rep.checkpoint else ""
    tiles = ""
    if rep.compare:
        def tile(label, a):
            return (f'<div class="tile"><div class="label">{label}</div><div class="value">{100 * a[0] / a[1]:.0f}%</div>'
                    f'<div class="sub">{a[0]} / {a[1]} cells match basic strategy</div></div>')
        tiles = ('<div class="tiles">' + tile("Overall agreement", (tot_ok, tot_n)) + tile("Hard totals", hard_a)
                 + tile("Soft totals", soft_a) + tile("Pairs", pair_a) + "</div>")
    legend = ('<div class="legend">' + "".join(
        f'<span><i class="sw {k}"></i>{k} = {v}</span>' for k, v in LETTER_NAMES.items())
        + (' <label><input type="checkbox" id="only-diff"> highlight differences only</label>' if rep.compare else "")
        + "</div>")
    bet_label = {"hilo": "Hi-Lo bet spread", "basic": "flat betting"}.get(rep.agent_name, f"{rep.agent_name.upper()} agent vs Hi-Lo reference")
    q_section = ""
    if rep.bet_q:
        what = ("Q-values per bet size (scaled reward units; outlined = chosen)" if rep.score_kind == "q"
                else "Policy probability per bet size (outlined = chosen)")
        q_section = (f'<h2 style="margin-top:18px">{what}</h2>' + _q_table(rep))
    body = f"""<div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="meta">{html.escape(rep.rules.describe())} &middot; bets {', '.join(f'{b:g}' for b in rep.bet_sizes)}{src}<br>
play tables assume true count {rep.true_count:+.1f} and {rep.decks_frac:.0%} of the shoe left; two-card hands, dealer up-card across the top</div>
{tiles}
{legend}
<div class="tables">
{_html_table("Hard totals", rep.hard_rows, rep.hard, rep.compare, hard_a)}
{_html_table("Soft totals", rep.soft_rows, rep.soft, rep.compare, soft_a)}
{_html_table("Pairs", rep.pair_rows, rep.pairs, rep.compare, pair_a)}
</div>
<div class="card"><h2>Bet size by true count — {bet_label}</h2>{_bet_svg(rep)}{q_section}</div>
<footer>generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by <code>blackjack-strategy</code> &middot; hover a cell for details</footer>
</div>"""
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}"
            f"<script>{_JS if rep.compare else ''}</script></body></html>")


# ---------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Show an agent's strategy tables and bet spread.")
    add_rules_args(p)
    p.add_argument("--agent", choices=["basic", "hilo", "dqn", "ppo", "rl"], default="rl",
                   help="rl = whatever kind the checkpoint holds")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--true-count", type=float, default=0.0, help="true count to assume for the play tables")
    p.add_argument("--decks-frac", type=float, default=0.6, help="fraction of shoe remaining to assume")
    p.add_argument("--no-compare", action="store_true", help="don't diff against basic strategy")
    p.add_argument("--html", type=str, default=None, metavar="PATH", help="also write a self-contained HTML report")
    p.add_argument("--open", action="store_true", help="open the HTML report in the browser (implies --html)")
    p.add_argument("--quiet", action="store_true", help="don't print the text tables")
    a = p.parse_args(argv)

    env = env_from_args(a)
    agent = make_agent(a.agent, env, a.checkpoint)
    rep = build_report(a.agent, agent, env, a.true_count, a.decks_frac, a.checkpoint, compare=not a.no_compare)
    if not a.quiet:
        print(render_text(rep))

    path = a.html
    if a.open and not path:
        path = f"strategy_{rep.agent_name}.html"
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_html(rep))
        print(f"\nHTML report written to {path}")
        if a.open:
            webbrowser.open("file://" + os.path.abspath(path))


if __name__ == "__main__":
    main()
