let decks = {
  llms,
  diffusion
};

let currentDeck = "llms";
let currentIndex = 0;
let bookmarks = {};  // { deckName: [index1, index2] }
let showBookmarkedOnly = false;

const deckSelect = document.getElementById("deckSelect");
const questionEl = document.getElementById("question");
const answerEl = document.getElementById("answer");
const counterEl = document.getElementById("counter");
const bookmarkEl = document.getElementById("bookmarkLabel");
const toggleBookmarkViewBtn = document.getElementById("toggleBookmarkView");

deckSelect.addEventListener("change", () => {
  currentDeck = deckSelect.value;
  currentIndex = 0;
  renderCard();
});

function getVisibleDeck() {
  const fullDeck = decks[currentDeck];
  const marked = bookmarks[currentDeck] || [];
  return showBookmarkedOnly
    ? marked.map(i => fullDeck[i]).filter(Boolean)
    : fullDeck;
}

function renderCard() {
  const visibleDeck = getVisibleDeck();
  if (visibleDeck.length === 0) {
    questionEl.textContent = "No bookmarked cards in this deck.";
    answerEl.textContent = "";
    counterEl.textContent = "";
    bookmarkEl.textContent = "";
    return;
  }

  const actualIndex = showBookmarkedOnly
    ? bookmarks[currentDeck][currentIndex]
    : currentIndex;

  const card = decks[currentDeck][actualIndex];
  questionEl.textContent = card.q;
  answerEl.textContent = card.a;
  answerEl.style.display = "none";
  counterEl.textContent = `Card ${currentIndex + 1} of ${visibleDeck.length}`;
  const isBookmarked = (bookmarks[currentDeck] || []).includes(actualIndex);
  bookmarkEl.textContent = isBookmarked ? "🔖 Bookmarked" : "";
}

function nextCard() {
  const visibleDeck = getVisibleDeck();
  currentIndex = (currentIndex + 1) % visibleDeck.length;
  renderCard();
}

function prevCard() {
  const visibleDeck = getVisibleDeck();
  currentIndex = (currentIndex - 1 + visibleDeck.length) % visibleDeck.length;
  renderCard();
}

function toggleAnswer() {
  answerEl.style.display = answerEl.style.display === "none" ? "block" : "none";
}

function toggleDarkMode() {
  document.body.classList.toggle("dark");
}

function toggleBookmark() {
  bookmarks[currentDeck] = bookmarks[currentDeck] || [];
  const actualIndex = showBookmarkedOnly
    ? bookmarks[currentDeck][currentIndex]
    : currentIndex;

  const idx = bookmarks[currentDeck].indexOf(actualIndex);
  if (idx === -1) {
    bookmarks[currentDeck].push(actualIndex);
  } else {
    bookmarks[currentDeck].splice(idx, 1);
  }
  renderCard();
}

function toggleBookmarkView() {
  showBookmarkedOnly = !showBookmarkedOnly;
  currentIndex = 0;
  toggleBookmarkViewBtn.textContent = showBookmarkedOnly
    ? "📄 Show All"
    : "🔖 Show Bookmarked";
  renderCard();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") nextCard();
  else if (e.key === "ArrowLeft") prevCard();
  else if (e.key.toLowerCase() === "a") toggleAnswer();
  else if (e.key.toLowerCase() === "b") toggleBookmark();
  else if (e.key.toLowerCase() === "d") toggleDarkMode();
  else if (e.key.toLowerCase() === "v") toggleBookmarkView();
});

renderCard();