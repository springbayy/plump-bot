const setup = document.querySelector("#setup");
const table = document.querySelector("#table");
const form = document.querySelector("#setup-form");
const opponentsInput = document.querySelector("#opponents");
const handSizeInput = document.querySelector("#hand-size");
const bidPositionInput = document.querySelector("#bid-position");
const newRoundButton = document.querySelector("#new-round");

let state = null;
let botAdvanceTimer = null;
let advancingBot = false;

function maxHandSize() {
  const players = Number(opponentsInput.value) + 1;
  return Math.min(10, Math.floor(52 / players));
}

function refreshSetupBounds() {
  const players = Number(opponentsInput.value) + 1;
  handSizeInput.max = String(maxHandSize());
  handSizeInput.min = "3";
  if (Number(handSizeInput.value) > maxHandSize()) {
    handSizeInput.value = String(maxHandSize());
  }
  bidPositionInput.innerHTML = "";
  for (let i = 1; i <= players; i += 1) {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${ordinal(i)} to bid`;
    bidPositionInput.append(option);
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  try {
    state = await postJson("/api/new", {
      opponents: Number(data.get("opponents")),
      hand_size: Number(data.get("hand_size")),
      bid_position: Number(data.get("bid_position")),
      seed: data.get("seed"),
    });
    setup.classList.add("hidden");
    table.classList.remove("hidden");
    render();
  } catch (error) {
    showSetupError(error.message);
  }
});

newRoundButton.addEventListener("click", () => {
  table.classList.add("hidden");
  setup.classList.remove("hidden");
});

opponentsInput.addEventListener("change", refreshSetupBounds);

function render() {
  if (!state) return;
  document.querySelector("#phase-title").textContent = phaseTitle(state.phase, state.done);
  document.querySelector("#trump-pill").textContent = `Trump: ${state.trump_label}`;
  document.querySelector("#hand-pill").textContent = `${state.hand_size} cards`;
  document.querySelector("#turn-pill").textContent = state.done
    ? "Round over"
    : state.human_turn
      ? "Your turn"
      : `${playerName(state.current_player)} thinking`;

  renderPlayers();
  renderTrick();
  renderActions();
  renderHand();
  renderLog();
  maybeScheduleBotAdvance();
}

function renderPlayers() {
  const root = document.querySelector("#players");
  root.innerHTML = "";
  for (const player of state.players) {
    const item = document.createElement("article");
    item.className = `player${player.is_current ? " current" : ""}`;
    item.innerHTML = `
      <div class="player-top">
        <span class="player-name">${escapeHtml(player.name)}</span>
        <span class="badge">${player.cards} cards</span>
      </div>
      <div class="player-stats">
        <span>Bid: <strong>${player.bid ?? "—"}</strong></span>
        <span>Tricks: <strong>${player.tricks}</strong></span>
        <span>Score: <strong>${player.score}</strong></span>
      </div>
      ${predictionHtml(player)}
    `;
    root.append(item);
  }
}

function renderTrick() {
  const root = document.querySelector("#trick");
  root.innerHTML = "";
  const trick = state.current_trick;
  const led = trick?.led_suit ? suitTitle(trick.led_suit) : "waiting for lead";
  document.querySelector("#leader-label").textContent = trick ? `Leader: ${playerName(trick.leader)} · Led: ${led}` : "";
  const players = trick ? trickOrder(trick.leader, state.players.length) : state.players.map((player) => player.id);
  for (const player of players) {
    const play = trick?.plays.find((row) => row.player === player);
    const won = trick?.winner === player;
    const slot = document.createElement("div");
    slot.className = `trick-slot${won ? " winner" : ""}`;
    slot.innerHTML = `
      <div class="slot-name">${play ? `${play.position + 1}. ` : ""}${playerName(player)}</div>
      ${play ? cardHtml(play.card) : '<div class="card empty">·</div>'}
      ${won ? '<div class="winner-label">Winner</div>' : ""}
    `;
    root.append(slot);
  }
}

function renderActions() {
  const root = document.querySelector("#action-panel");
  root.innerHTML = "";
  if (state.done) {
    const scores = state.players.map((p) => `${p.name}: ${p.score}`).join(" · ");
    root.innerHTML = `<h2>Round Complete</h2><p class="subtle">${scores}</p>`;
    return;
  }
  if (!state.human_turn) {
    root.innerHTML = `<h2>Waiting</h2><p class="subtle">${playerName(state.current_player)} is to act.</p>`;
    return;
  }
  if (state.phase === "bidding") {
    root.innerHTML = `<h2>Your Bid</h2><p class="subtle">Choose a legal bid. The forbidden last bid is omitted automatically.</p>`;
    const buttons = document.createElement("div");
    buttons.className = "bid-buttons";
    for (const bid of state.legal_bids) {
      const button = document.createElement("button");
      button.className = "bid-button";
      button.textContent = bid;
      button.addEventListener("click", () => submitBid(bid));
      buttons.append(button);
    }
    root.append(buttons);
    return;
  }
  if (state.phase === "playing") {
    const ledSuit = state.current_trick?.led_suit ? suitTitle(state.current_trick.led_suit) : null;
    const hint = ledSuit
      ? `Led suit is ${ledSuit}. If you have none, every card is legal.`
      : "You are leading and may play any card.";
    root.innerHTML = `<h2>Your Play</h2><p class="subtle">${hint}</p>`;
  }
}

function renderHand() {
  const root = document.querySelector("#hand");
  root.innerHTML = "";
  document.querySelector("#legal-hint").textContent =
    state.phase === "playing" && state.human_turn ? "Click a legal card" : "Always visible";
  for (const card of state.my_hand) {
    const button = document.createElement("button");
    button.className = `card-button${card.legal && state.human_turn && state.phase === "playing" ? " legal" : ""}`;
    button.disabled = !(card.legal && state.human_turn && state.phase === "playing");
    button.innerHTML = cardHtml(card);
    button.addEventListener("click", () => submitPlay(card));
    root.append(button);
  }
}

function renderLog() {
  const messages = document.querySelector("#messages");
  messages.innerHTML = "";
  for (const message of state.messages.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "message";
    row.textContent = message;
    messages.append(row);
  }

  const events = document.querySelector("#events");
  events.innerHTML = "";
  for (const event of state.event_log.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "event";
    row.textContent = eventText(event);
    events.append(row);
  }
}

async function submitBid(bid) {
  try {
    state = await postJson("/api/bid", { bid });
    render();
  } catch (error) {
    showActionError(error.message);
  }
}

async function submitPlay(card) {
  try {
    state = await postJson("/api/play", { suit: card.suit, rank: card.rank });
    render();
  } catch (error) {
    showActionError(error.message);
  }
}

function maybeScheduleBotAdvance() {
  if (!state || state.done || state.human_turn || advancingBot) return;
  if (botAdvanceTimer) return;
  const delay = state.phase === "playing" ? 700 : 300;
  botAdvanceTimer = window.setTimeout(advanceBot, delay);
}

async function advanceBot() {
  botAdvanceTimer = null;
  if (!state || state.done || state.human_turn || advancingBot) return;
  advancingBot = true;
  try {
    state = await postJson("/api/advance", {});
    render();
  } catch (error) {
    showActionError(error.message);
  } finally {
    advancingBot = false;
    maybeScheduleBotAdvance();
  }
}

function predictionHtml(player) {
  const prediction = player.prediction;
  if (!prediction?.rows?.length) return "";
  const self = prediction.rows.find((row) => row.player === player.id);
  const point = self ? `${Math.round(self.point_prob * 100)}%` : "—";
  const rows = prediction.rows
    .map((row) => {
      const name = playerName(row.player);
      const expected = Number(row.expected_tricks).toFixed(1);
      const top = `${row.top_tricks}`;
      const confidence = Math.round(row.top_tricks_prob * 100);
      const pointChance = Math.round(row.point_prob * 100);
      return `
        <div class="prediction-row">
          <span>${escapeHtml(name)}</span>
          <strong>${expected}</strong>
          <span>${top} @ ${confidence}% · P${pointChance}%</span>
        </div>
      `;
    })
    .join("");
  return `
    <div class="prediction-box">
      <div class="prediction-head">
        <span>Model POV</span>
        <strong>P points ${point}</strong>
      </div>
      <div class="prediction-grid">
        ${rows}
      </div>
    </div>
  `;
}

function cardHtml(card) {
  const isTrump = state?.trump && card.suit === state.trump;
  return `
    <div class="card ${card.color}${isTrump ? " trump-card" : ""}">
      <span>${escapeHtml(card.label)}</span>
      ${isTrump ? '<small>trump</small>' : ""}
    </div>
  `;
}

function eventText(event) {
  if (event.type === "bid") return `${playerName(event.player)} bid ${event.bid}`;
  if (event.type === "play") return `${playerName(event.player)} played ${event.card.label}`;
  if (event.type === "trick_win") return `${playerName(event.player)} won trick ${event.trick_index + 1}`;
  if (event.type === "round_start") return "Round started";
  if (event.type === "round_end") return "Round ended";
  return event.type;
}

function phaseTitle(phase, done) {
  if (done) return "Round Over";
  if (phase === "bidding") return "Bidding";
  if (phase === "playing") return "Playing";
  return phase;
}

function playerName(playerId) {
  if (playerId === null || playerId === undefined) return "Nobody";
  return playerId === 0 ? "You" : `Model ${playerId}`;
}

function trickOrder(leader, playerCount) {
  return Array.from({ length: playerCount }, (_, index) => (leader + index) % playerCount);
}

function suitTitle(suit) {
  return {
    spades: "Spades",
    hearts: "Hearts",
    diamonds: "Diamonds",
    clubs: "Clubs",
  }[suit] || suit;
}

function ordinal(value) {
  const suffix = value % 100 >= 10 && value % 100 <= 20 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[value % 10] || "th";
  return `${value}${suffix}`;
}

function showSetupError(message) {
  let error = setup.querySelector(".error");
  if (!error) {
    error = document.createElement("div");
    error.className = "error";
    form.prepend(error);
  }
  error.textContent = message;
}

function showActionError(message) {
  const root = document.querySelector("#action-panel");
  const error = document.createElement("div");
  error.className = "error";
  error.textContent = message;
  root.append(error);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

refreshSetupBounds();
