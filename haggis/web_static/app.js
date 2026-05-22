let sessionId = null;
let state = null;
let selected = new Set();
let handOrder = [];
let draggedCardKey = null;
let dropTarget = null;

const $ = (id) => document.getElementById(id);

async function api(path, payload = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, ...payload }),
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || 'Request failed');
  return data;
}

function normalizeState(rawState) {
  const normalized = {
    targetScore: 350,
    handNumber: 1,
    cumulativeScore: { human: 0, cpu: 0 },
    gameWinner: null,
    cpuWilds: [],
    lastPlayedCards: [],
    currentPlayCards: [],
    currentPlayCleared: false,
    handScoreBreakdown: null,
    trickCards: [],
    ...rawState,
  };
  normalized.cpuWilds = visibleCpuWilds(normalized);
  if ((!normalized.currentPlayCards || normalized.currentPlayCards.length === 0) && normalized.lastPlayedCards && normalized.lastPlayedCards.length > 0) {
    normalized.currentPlayCards = normalized.lastPlayedCards;
  }
  if ((!normalized.trickCards || normalized.trickCards.length === 0) && normalized.currentPlayCards && normalized.currentPlayCards.length > 0) {
    normalized.trickCards = normalized.currentPlayCards;
  }
  syncHandOrder(normalized.humanHand || []);
  return normalized;
}

function visibleCpuWilds(currentState) {
  if (currentState.cpuWilds && currentState.cpuWilds.length > 0) return currentState.cpuWilds;
  if (!currentState.bettingComplete && currentState.cpuCards >= 3) return [
    { key: 'JSHADOWW', name: 'J*', rank: 'J', suit: 'S', points: 2, wild: true },
    { key: 'QSHADOWW', name: 'Q*', rank: 'Q', suit: 'S', points: 3, wild: true },
    { key: 'KSHADOWW', name: 'K*', rank: 'K', suit: 'S', points: 5, wild: true },
  ];
  return [];
}

function syncHandOrder(hand) {
  const handKeys = new Set(hand.map((card) => card.key));
  handOrder = handOrder.filter((key) => handKeys.has(key));
  for (const card of hand) {
    if (!handOrder.includes(card.key)) handOrder.push(card.key);
  }
}


async function newGame() {
  selected.clear();
  handOrder = [];
  const data = await api('/api/new', { targetScore: 350 });
  sessionId = data.sessionId;
  state = normalizeState(data.state);
  render();
}

async function bet(amount) {
  const data = await api('/api/bet', { amount });
  state = normalizeState(data.state);
  selected.clear();
  render();
}

async function playSelected() {
  const data = await api('/api/play', { cards: [...selected] });
  state = normalizeState(data.state);
  selected.clear();
  render();
}

async function passTurn() {
  const data = await api('/api/pass');
  state = normalizeState(data.state);
  selected.clear();
  render();
}

async function nextHand() {
  if (!state || !state.handWinner) {
    alert('Finish the current hand before starting the next one.');
    return;
  }
  handOrder = [];
  const data = await api('/api/next-hand');
  state = normalizeState(data.state);
  selected.clear();
  render();
}

function render() {
  if (!state) return;
  $('bet-panel').style.display = !state.bettingComplete && !state.gameWinner ? 'block' : 'none';
  $('game-status').textContent = `Game to ${state.targetScore}: You ${state.cumulativeScore.human} — CPU ${state.cumulativeScore.cpu}`;
  $('target-score').textContent = state.targetScore;
  $('hand-number').textContent = state.handNumber;
  $('game-score-human').textContent = state.cumulativeScore.human;
  $('game-score-cpu').textContent = state.cumulativeScore.cpu;
  const canStartNextHand = Boolean(state.handWinner && !state.gameWinner);
  $('next-hand').style.display = canStartNextHand ? 'inline-block' : 'none';
  $('next-hand').disabled = !canStartNextHand;
  $('cpu-cards').textContent = state.cpuCards;
  $('cpu-wilds').innerHTML = state.cpuWilds.length ? `<p class="section-label">Visible wilds</p>${state.cpuWilds.map((card) => cardHtml(card, 'mini-card')).join('')}` : '<p class="section-label muted">No CPU wilds remaining</p>';
  $('cpu-hand').innerHTML = Array.from({ length: Math.max(0, state.cpuCards - state.cpuWilds.length) }, () => '<span class="card-back"></span>').join('');
  $('turn').textContent = state.gameWinner ? `${state.gameWinner === 'human' ? 'You win' : 'CPU wins'} the game — ${state.cumulativeScore.human}–${state.cumulativeScore.cpu}` : state.handWinner ? `${state.handWinner === 'human' ? 'You win' : 'CPU wins'} the hand — score ${state.score.human}–${state.score.cpu}` : `${state.currentPlayer === 'human' ? 'Your' : 'CPU'} turn`;
  $('last-play').textContent = state.lastCombination ? `${state.lastPlayer === 'human' ? 'You' : 'CPU'} played ${state.lastCombination}` : 'No current trick.';
  $('trick-message').textContent = state.currentPlayCleared ? 'Stack cleared.' : '';
  $('last-played-cards').innerHTML = state.lastPlayedCards.length ? state.lastPlayedCards.map((card) => cardHtml(card, 'mini-card')).join('') : '<p class="section-label muted">No cards on the current trick</p>';
  $('trick-cards').innerHTML = trickCardsHtml();
  $('selected-cards').textContent = selectedCardNames();
  $('points').textContent = `Captured: You ${state.capturedPoints.human} · CPU ${state.capturedPoints.cpu}. Trick points: ${state.trickPoints}. Bets: You ${state.bets.human} · CPU ${state.bets.cpu}.`;
  $('score-breakdown').innerHTML = scoreBreakdownHtml();
  $('hint').textContent = state.selectedHint;
  renderHand();
  $('play-selected').disabled = !state.bettingComplete || state.currentPlayer !== 'human' || selected.size === 0 || Boolean(state.handWinner);
  $('pass').disabled = !state.canPass || state.currentPlayer !== 'human' || Boolean(state.handWinner);
  $('move-count').textContent = state.legalMoveCount;
  $('legal-moves').innerHTML = state.legalMoves.map((move) => `<li>${escapeHtml(move.label)}</li>`).join('');
  $('log').innerHTML = state.log.map((line) => `<li>${escapeHtml(line)}</li>`).join('');
}

function trickCardsHtml() {
  if (state.currentPlayCleared) return '';
  const cards = state.currentPlayCards && state.currentPlayCards.length ? state.currentPlayCards : state.lastPlayedCards;
  return cards && cards.length ? cards.map((card) => cardHtml(card, 'mini-card')).join('') : '';
}

function scoreBreakdownHtml() {
  const breakdown = state.handScoreBreakdown;
  if (!breakdown) return '';
  const rows = breakdown.rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.label)}</td>
      <td>${row.human}</td>
      <td>${row.cpu}</td>
    </tr>
  `).join('');
  return `
    <table>
      <thead><tr><th>Source</th><th>You</th><th>CPU</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th>Total</th><th>${breakdown.total.human}</th><th>${breakdown.total.cpu}</th></tr></tfoot>
    </table>
  `;
}

function selectedCardNames() {
  if (!state || selected.size === 0) return '—';
  return orderedHumanHand().filter((card) => selected.has(card.key)).map((card) => card.name).join(' ');
}

function orderedHumanHand() {
  if (!state) return [];
  const byKey = new Map(state.humanHand.map((card) => [card.key, card]));
  return handOrder.map((key) => byKey.get(key)).filter(Boolean);
}

function renderHand() {
  $('human-hand').innerHTML = '';
  for (const card of orderedHumanHand()) {
    const button = document.createElement('button');
    button.className = `card ${cardColorClass(card)} ${selected.has(card.key) ? 'selected' : ''} ${draggedCardKey === card.key ? 'dragging' : ''} ${dropClassFor(card.key)}`;
    button.innerHTML = cardFaceHtml(card);
    button.type = 'button';
    button.draggable = true;
    button.disabled = Boolean(state.handWinner);
    button.dataset.cardKey = card.key;
    button.addEventListener('click', () => {
      if (!state.bettingComplete || state.currentPlayer !== 'human' || state.handWinner) return;
      if (selected.has(card.key)) selected.delete(card.key);
      else selected.add(card.key);
      render();
    });
    button.addEventListener('dragstart', (event) => {
      draggedCardKey = card.key;
      dropTarget = null;
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', card.key);
      button.classList.add('dragging');
    });
    button.addEventListener('dragover', (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
      const side = dropSide(event, button);
      const nextTarget = { key: card.key, side };
      if (!dropTarget || dropTarget.key !== nextTarget.key || dropTarget.side !== nextTarget.side) {
        dropTarget = nextTarget;
        renderHand();
      }
    });
    button.addEventListener('dragleave', (event) => {
      if (!button.contains(event.relatedTarget)) {
        dropTarget = null;
        renderHand();
      }
    });
    button.addEventListener('drop', (event) => {
      event.preventDefault();
      const sourceKey = draggedCardKey || event.dataTransfer.getData('text/plain');
      moveCard(sourceKey, card.key, dropTarget?.side || dropSide(event, button));
      draggedCardKey = null;
      dropTarget = null;
      render();
    });
    button.addEventListener('dragend', () => {
      draggedCardKey = null;
      dropTarget = null;
      render();
    });
    $('human-hand').appendChild(button);
  }
}

function dropSide(event, element) {
  const rect = element.getBoundingClientRect();
  return event.clientX < rect.left + rect.width / 2 ? 'before' : 'after';
}

function dropClassFor(cardKey) {
  if (!dropTarget || dropTarget.key !== cardKey || draggedCardKey === cardKey) return '';
  return dropTarget.side === 'after' ? 'drop-after' : 'drop-before';
}

function moveCard(sourceKey, targetKey, side = 'before') {
  if (!sourceKey || !targetKey || sourceKey === targetKey) return;
  const withoutSource = handOrder.filter((key) => key !== sourceKey);
  const targetIndex = withoutSource.indexOf(targetKey);
  if (targetIndex === -1) return;
  withoutSource.splice(side === 'after' ? targetIndex + 1 : targetIndex, 0, sourceKey);
  handOrder = withoutSource;
}

function sortByRank() {
  handOrder = orderedHumanHand()
    .slice()
    .sort((a, b) => cardRankValue(a) - cardRankValue(b) || suitValue(a) - suitValue(b) || Number(a.wild) - Number(b.wild))
    .map((card) => card.key);
  render();
}

function sortBySuit() {
  handOrder = orderedHumanHand()
    .slice()
    .sort((a, b) => suitValue(a) - suitValue(b) || cardRankValue(a) - cardRankValue(b) || Number(a.wild) - Number(b.wild))
    .map((card) => card.key);
  render();
}

function cardRankValue(card) {
  return { J: 11, Q: 12, K: 13 }[card.rank] || Number(card.rank);
}

function suitValue(card) {
  return { C: 0, D: 1, H: 2, S: 3 }[card.suit] ?? 4;
}

function cardHtml(card, className = 'card') {
  return `<span class="${className} ${cardColorClass(card)}">${cardFaceHtml(card)}</span>`;
}

function cardFaceHtml(card) {
  return `<span>${escapeHtml(card.name)}</span><span class="points">${card.points ? `${card.points}pt` : ''}</span><span class="wild">${card.wild ? 'wild' : ''}</span>`;
}

function cardColorClass(card) {
  return ['D', 'H'].includes(card.suit) ? 'red' : '';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char]));
}

document.addEventListener('click', async (event) => {
  try {
    const betButton = event.target.closest('[data-bet]');
    if (betButton) await bet(Number(betButton.dataset.bet));
  } catch (error) { alert(error.message); }
});
$('new-game').addEventListener('click', () => newGame().catch((error) => alert(error.message)));
$('sort-rank').addEventListener('click', sortByRank);
$('sort-suit').addEventListener('click', sortBySuit);
$('next-hand').addEventListener('click', () => nextHand().catch((error) => alert(error.message)));
$('play-selected').addEventListener('click', () => playSelected().catch((error) => alert(error.message)));
$('pass').addEventListener('click', () => passTurn().catch((error) => alert(error.message)));

newGame().catch((error) => alert(error.message));
