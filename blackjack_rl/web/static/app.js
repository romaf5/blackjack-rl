/* Blackjack RL — browser front-end for the Gymnasium env served by blackjack_rl.web.server */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let state = null;
let advice = null;
let seenCards = new Set();
let auto = { on: false, timer: null };
let showCount = true;
let showAdvice = true;
const DEAL_STAGGER = 260;   // ms between consecutive dealt cards
const DEAL_DURATION = 520;  // ms for one card to land (keep in sync with style.css)
let renderToken = 0;

// ---------------------------------------------------------------- api
async function api(path, body) {
  // GET when no body is given; POST (JSON) otherwise. Pass {} to POST without arguments.
  const opts = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* ignore */ }
  if (!res.ok || (data && data.error)) throw new Error((data && data.error) || res.statusText);
  return data;
}

async function refresh(newState) {
  state = newState || await api("/api/state");
  advice = state.phase === "done" ? null : await api("/api/advice");
  render();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 2600);
}

const fmt = (n, sign = false) => {
  const s = Math.abs(n) % 1 === 0 ? String(Math.abs(n)) : Math.abs(n).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return (n < 0 ? "−" : (sign && n > 0 ? "+" : "")) + s;
};

// ---------------------------------------------------------------- rendering
function cardEl(c, key, isNew, flip) {
  const el = document.createElement("div");
  if (c.hidden) {
    el.className = "card back";
  } else {
    el.className = "card" + (c.red ? " red" : "");
    el.innerHTML = `<div class="corner">${c.rank}<small>${c.suit}</small></div>` +
                   `<div class="pip">${c.suit}</div>` +
                   `<div class="corner br">${c.rank}<small>${c.suit}</small></div>`;
  }
  if (isNew) el.classList.add(flip ? "flip" : "deal");
  return el;
}

function renderCards(container, cards, prefix, newKeys, newCards) {
  container.innerHTML = "";
  cards.forEach((c, i) => {
    const key = `${prefix}:${i}:${c.hidden ? "X" : c.rank + c.suit}`;
    const isNew = !seenCards.has(key);
    const wasHidden = seenCards.has(`${prefix}:${i}:X`) && !c.hidden;
    const el = cardEl(c, key, isNew, wasHidden);
    if (isNew) newCards.push({ el, prefix, idx: i, flip: wasHidden });
    container.appendChild(el);
    newKeys.push(key);
  });
}

// Assign animation delays so cards land one after another, in casino dealing order
// (player, dealer up-card, player, dealer hole card; everything else in the order it appears).
// Returns the total time until the last card has landed.
function scheduleDeals(newCards) {
  const initial = { "p0:0": 0, "d:0": 1, "p0:1": 2, "d:1": 3 };
  const ordered = newCards.slice().sort((a, b) => {
    const ka = initial[`${a.prefix}:${a.idx}`], kb = initial[`${b.prefix}:${b.idx}`];
    if (ka !== undefined && kb !== undefined) return ka - kb;
    if (ka !== undefined) return -1;
    if (kb !== undefined) return 1;
    return 0;
  });
  let t = 0;
  ordered.forEach((c) => {
    c.el.style.animationDelay = t + "ms";
    t += c.flip ? DEAL_STAGGER + 200 : DEAL_STAGGER;
  });
  return ordered.length ? t - DEAL_STAGGER + DEAL_DURATION : 0;
}

function chipClass(i) { return "c" + Math.min(i, 6); }

function render() {
  const s = state;
  if (!s) return;
  const newKeys = [];
  const newCards = [];
  const token = ++renderToken;
  if (s.phase === "bet") seenCards.clear();

  // header
  $("#rules").textContent = s.rules;
  const rd = s.rules_dict || {};
  const payout = rd.blackjack_payout === 1.5 ? "3 TO 2" : rd.blackjack_payout === 1.2 ? "6 TO 5" : `${rd.blackjack_payout} TO 1`;
  $("#table").dataset.felt = `BLACKJACK PAYS ${payout}  ·  DEALER ${rd.dealer_hits_soft_17 ? "HITS SOFT 17" : "STANDS ON ALL 17s"}`;
  $("#bankroll").textContent = fmt(s.bankroll);
  const net = s.bankroll - s.start_bankroll;
  const netEl = $("#net");
  netEl.textContent = fmt(net, true);
  netEl.className = "value " + (net > 0 ? "pos" : net < 0 ? "neg" : "");
  $("#rounds").textContent = s.rounds;

  // dealer
  const dt = $("#dealer-total");
  if (s.dealer) {
    renderCards($("#dealer-cards"), s.dealer.cards, "d", newKeys, newCards);
    if (s.dealer.total != null) {
      dt.textContent = (s.dealer.blackjack ? "Blackjack" : (s.dealer.soft ? "soft " : "") + s.dealer.total) + (s.dealer.bust ? " · bust" : "");
      dt.className = "total-badge reveal" + (s.dealer.bust ? " bust" : s.dealer.blackjack ? " bj" : "");
    } else {
      dt.className = "total-badge hidden";
    }
  } else {
    $("#dealer-cards").innerHTML = "";
    dt.className = "total-badge hidden";
  }

  // player hands
  const hands = $("#hands");
  hands.innerHTML = "";
  s.hands.forEach((h, i) => {
    const el = document.createElement("div");
    el.className = "hand" + (h.active ? " active" : "");
    if (h.result) {
      const p = h.result.profit;
      const cls = h.blackjack && p > 0 ? "bj" : p > 0 ? "win" : p < 0 ? "lose" : "push";
      const label = h.blackjack && p > 0 ? "BLACKJACK" : p > 0 ? "WIN" : p < 0 ? (h.surrendered ? "SURRENDER" : h.bust ? "BUST" : "LOSE") : "PUSH";
      el.innerHTML = `<div class="result-tag ${cls}">${label} ${p !== 0 ? fmt(p, true) : ""}</div>`;
    }
    const cards = document.createElement("div");
    cards.className = "cards";
    renderCards(cards, h.cards, "p" + i, newKeys, newCards);
    el.appendChild(cards);
    const meta = document.createElement("div");
    meta.className = "meta";
    const badgeCls = h.bust ? " bust" : h.blackjack ? " bj" : "";
    const tags = [];
    if (h.is_split) tags.push("split");
    if (h.doubled) tags.push("doubled");
    if (h.surrendered) tags.push("surrendered");
    const chipIdx = Math.max(0, s.bet_sizes.indexOf(h.doubled ? h.bet / 2 : h.bet));
    meta.innerHTML = `<span class="total-badge${badgeCls}">${h.blackjack ? "BJ" : (h.soft ? "soft " : "") + h.total}</span>` +
                     `<span class="tag">${tags.join(" · ")}</span>` +
                     `<span class="chip mini ${chipClass(chipIdx)}"><span>${fmt(h.bet)}</span></span>`;
    el.appendChild(meta);
    hands.appendChild(el);
  });

  // bet prompt
  const betPrompt = $("#bet-prompt");
  if (s.phase === "bet") {
    betPrompt.classList.remove("hidden");
    const chips = $("#chips");
    chips.innerHTML = "";
    s.bet_sizes.forEach((b, i) => {
      const btn = document.createElement("button");
      btn.className = "chip " + chipClass(i) + (showAdvice && advice && advice.hilo_bet_index === i ? " suggest" : "");
      btn.innerHTML = `<span>${fmt(b)}</span>`;
      btn.title = `bet ${b} (key ${i + 1})`;
      btn.onclick = () => userBet(i);
      chips.appendChild(btn);
    });
    $("#bet-hint").textContent = (showAdvice && advice && advice.hilo_bet != null)
      ? `Hi-Lo suggests ${fmt(advice.hilo_bet)}` + (advice.dqn && advice.dqn.best ? ` · DQN would ${advice.dqn.best}` : "") : "";
  } else {
    betPrompt.classList.add("hidden");
  }

  // actions
  const actions = $("#actions");
  if (s.phase === "play") {
    actions.classList.remove("hidden");
    $$("#actions .act").forEach((b) => {
      const a = b.dataset.action;
      b.disabled = !s.legal.includes(a);
      b.classList.toggle("suggested", showAdvice && advice && advice.basic === a);
    });
  } else {
    actions.classList.add("hidden");
  }

  // banner
  const banner = $("#banner");
  if (s.phase === "done" && s.last) {
    const p = s.last.profit;
    const anyBJ = s.hands.some((h) => h.blackjack && h.result && h.result.profit > 0);
    const main = $("#banner-main");
    if (p > 0) { main.textContent = (anyBJ ? "BLACKJACK!  " : "YOU WIN  ") + fmt(p, true); main.className = "banner-main " + (anyBJ ? "bj" : "win"); }
    else if (p < 0) { main.textContent = "YOU LOSE  " + fmt(p, true); main.className = "banner-main lose"; }
    else { main.textContent = "PUSH"; main.className = "banner-main push"; }
    const sub = s.last.results.map((r, i) => (s.last.results.length > 1 ? `hand ${i + 1}: ` : "") + `${r.label} (${fmt(r.profit, true)})`).join("  ·  ");
    $("#banner-sub").textContent = sub + (s.last.shuffled ? "  ·  (shoe was shuffled before this round)" : "");
    banner.classList.add("hidden");
    const wait = scheduleDeals(newCards) + 150;
    setTimeout(() => { if (token === renderToken && state.phase === "done") banner.classList.remove("hidden"); }, wait);
    document.body.style.setProperty("--reveal-delay", wait + "ms");
  } else {
    banner.classList.add("hidden");
    scheduleDeals(newCards);
    document.body.style.setProperty("--reveal-delay", "0ms");
  }

  // count panel
  const c = s.count;
  $("#rc").textContent = fmt(c.running, true);
  $("#tc").textContent = (c.true >= 0 ? "+" : "−") + Math.abs(c.true).toFixed(1);
  $("#tc").className = c.true >= 2 ? "pos" : c.true <= -2 ? "neg" : "";
  $("#decks").textContent = c.decks_remaining.toFixed(1);
  $("#shoe-fill").style.width = (100 * c.cards_dealt / c.total_cards).toFixed(1) + "%";
  $("#cut-mark").style.left = (100 * c.cut_card / c.total_cards).toFixed(1) + "%";
  $("#shoe-text").textContent = `${c.cards_dealt} / ${c.total_cards} cards dealt · cut card at ${Math.round(100 * c.penetration)}%`;

  // advisor
  const advBasic = $("#adv-basic"), advHilo = $("#adv-hilo"), advDqn = $("#adv-dqn"), qbars = $("#qbars");
  qbars.innerHTML = "";
  if (advice && s.phase !== "done") {
    advBasic.textContent = advice.basic ? advice.basic.toUpperCase() : "–";
    advHilo.textContent = advice.hilo_bet != null ? `bet ${fmt(advice.hilo_bet)}` : "–";
    if (advice.dqn) {
      advDqn.textContent = advice.dqn.best ? advice.dqn.best.toUpperCase() : "–";
      const qs = advice.dqn.q;
      const maxAbs = Math.max(0.05, ...qs.map((e) => Math.abs(e.q)));
      qs.forEach((e) => {
        const row = document.createElement("div");
        row.className = "qbar" + (e.action === advice.dqn.best ? " best" : "");
        const w = 50 * Math.abs(e.q) / maxAbs;
        const left = e.q >= 0 ? 50 : 50 - w;
        row.innerHTML = `<span>${e.action}</span><div class="track"><i class="${e.q < 0 ? "neg" : ""}" style="left:${left}%;width:${w}%"></i></div><span class="val">${fmt(e.q, true)}</span>`;
        qbars.appendChild(row);
      });
    } else {
      advDqn.textContent = "–";
    }
  } else {
    advBasic.textContent = advHilo.textContent = advDqn.textContent = "–";
  }
  $("#dqn-note").textContent = s.dqn.loaded ? `Q-values in bet units · ${s.dqn.checkpoint}` :
    (s.dqn.error ? `DQN not available: ${s.dqn.error}` : "No DQN checkpoint found — run blackjack-train, then restart with --checkpoint");

  // agent select
  const sel = $("#agent-select");
  const dqnOpt = sel.querySelector('option[value="dqn"]');
  dqnOpt.disabled = !s.dqn.loaded;
  if (dqnOpt.disabled && sel.value === "dqn") sel.value = "basic";

  // session
  $("#wlp").textContent = `${s.wins} / ${s.losses} / ${s.pushes}`;
  $("#winrate").textContent = s.rounds ? (100 * s.wins / s.rounds).toFixed(1) + "%" : "–";
  $("#shuffles").textContent = c.num_shuffles - 1;
  drawSpark(s.bankroll_history, s.start_bankroll);

  newKeys.forEach((k) => seenCards.add(k));
}

function drawSpark(hist, start) {
  const cv = $("#spark");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!hist || hist.length < 2) {
    ctx.fillStyle = "#4b5661"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("bankroll over time", W / 2, H / 2 + 4);
    return;
  }
  const lo = Math.min(start, ...hist), hi = Math.max(start, ...hist);
  const pad = 6, span = Math.max(hi - lo, 1e-9);
  const x = (i) => pad + (W - 2 * pad) * i / (hist.length - 1);
  const y = (v) => H - pad - (H - 2 * pad) * (v - lo) / span;
  ctx.setLineDash([3, 3]); ctx.strokeStyle = "#4b5661"; ctx.beginPath(); ctx.moveTo(pad, y(start)); ctx.lineTo(W - pad, y(start)); ctx.stroke();
  ctx.setLineDash([]);
  const last = hist[hist.length - 1];
  ctx.strokeStyle = last >= start ? "#48c774" : "#ef5b5b"; ctx.lineWidth = 2; ctx.beginPath();
  hist.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
}

// ---------------------------------------------------------------- user actions
async function userBet(i) {
  stopAuto();
  try { await refresh(await api("/api/bet", { index: i })); } catch (e) { toast(e.message); }
}
async function userAct(a) {
  stopAuto();
  try { await refresh(await api("/api/action", { action: a })); } catch (e) { toast(e.message); }
}
async function nextRound() {
  try { await refresh(await api("/api/next", {})); } catch (e) { toast(e.message); }
}

// ---------------------------------------------------------------- autoplay
function flashAction(name) {
  const btn = $(`#actions .act[data-action="${name}"]`);
  if (btn) { btn.classList.add("flash"); setTimeout(() => btn.classList.remove("flash"), 450); }
  const pill = $("#agent-flash");
  pill.textContent = `agent: ${name}`;
  pill.classList.remove("hidden");
  clearTimeout(flashAction._t);
  flashAction._t = setTimeout(() => pill.classList.add("hidden"), 900);
}

async function agentStep() {
  const st = await api("/api/agent_step", { agent: $("#agent-select").value });
  await refresh(st);
  if (st.agent_action) flashAction(st.agent_action);
  return st;
}

async function autoTick() {
  if (!auto.on) return;
  try {
    const st = await agentStep();
    const delay = +$("#speed").value;
    auto.timer = setTimeout(autoTick, st.phase === "done" ? Math.max(delay * 2, 350) : delay);
  } catch (e) { toast(e.message); stopAuto(); }
}
function startAuto() {
  auto.on = true;
  const b = $("#btn-auto"); b.textContent = "■ Stop"; b.classList.add("stop");
  autoTick();
}
function stopAuto() {
  if (!auto.on) return;
  auto.on = false;
  clearTimeout(auto.timer);
  const b = $("#btn-auto"); b.textContent = "▶ Start"; b.classList.remove("stop");
}

// ---------------------------------------------------------------- wiring
$$("#actions .act").forEach((b) => (b.onclick = () => userAct(b.dataset.action)));
$("#btn-next").onclick = nextRound;
$("#btn-auto").onclick = () => (auto.on ? stopAuto() : startAuto());
$("#btn-step").onclick = async () => { stopAuto(); try { await agentStep(); } catch (e) { toast(e.message); } };
$("#toggle-count").onchange = (e) => { showCount = e.target.checked; $("#count-body").classList.toggle("hidden", !showCount); };
$("#toggle-advice").onchange = (e) => { showAdvice = e.target.checked; $("#advice-body").classList.toggle("hidden", !showAdvice); render(); };
$("#btn-settings").onclick = () => $("#modal").classList.remove("hidden");
$("#btn-report").onclick = () => {
  const agent = $("#agent-select").value;
  const tc = state && state.count ? state.count.true : 0;
  window.open(`/report?agent=${encodeURIComponent(agent)}&tc=${encodeURIComponent(tc.toFixed(1))}`, "_blank");
};
$("#btn-cancel").onclick = () => $("#modal").classList.add("hidden");
$("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); };
$("#settings-form").onsubmit = async (e) => {
  e.preventDefault();
  stopAuto();
  const f = new FormData(e.target);
  const rules = {};
  ["num_decks", "penetration", "blackjack_payout", "max_splits"].forEach((k) => (rules[k] = f.get(k)));
  ["dealer_hits_soft_17", "double_after_split", "surrender", "dealer_peeks", "resplit_aces", "hit_split_aces"]
    .forEach((k) => (rules[k] = f.get(k) === "on"));
  rules.double_on = f.get("double_on") || null;
  const bets = String(f.get("bet_sizes")).split(",").map((x) => parseFloat(x)).filter((x) => !isNaN(x));
  try {
    seenCards.clear();
    await refresh(await api("/api/new", { rules, bet_sizes: bets, bankroll: f.get("bankroll"), seed: f.get("seed") }));
    $("#modal").classList.add("hidden");
  } catch (err) { toast(err.message); }
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !state) return;
  if (e.key === "Escape") { $("#modal").classList.add("hidden"); return; }
  if (state.phase === "bet" && /^[1-9]$/.test(e.key)) {
    const i = parseInt(e.key, 10) - 1;
    if (i < state.bet_sizes.length) userBet(i);
  } else if (state.phase === "play") {
    const map = { s: "stand", h: "hit", d: "double", p: "split", r: "surrender" };
    const a = map[e.key.toLowerCase()];
    if (a && state.legal.includes(a)) userAct(a);
  } else if (state.phase === "done" && (e.key === "Enter" || e.key === " ")) {
    e.preventDefault();
    nextRound();
  }
});

refresh().catch((e) => toast("Cannot reach the game server: " + e.message));
