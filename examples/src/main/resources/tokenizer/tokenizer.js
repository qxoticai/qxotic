const $ = (id) => document.getElementById(id);

const el = {
    tokenizer: $("tokenizer"), text: $("text"), clear: $("clear"),
    decodeBtn: $("decodeBtn"), copyTokens: $("copyTokens"), tokenIds: $("tokenIds"),
    decoded: $("decoded"), tokensStage: $("tokensStage"), tokensCard: $("tokensCard"),
    statsPills: $("statsPills"), status: $("status"),
    tokenCount: $("tokenCount"), charCount: $("charCount"), ratio: $("ratio"),
    theme: $("theme"), showIds: $("showIds"),
};

const N_COLORS = 8;
const THEME_KEY = "jota-tokenizer-theme";
let timer = null;

function status(msg, err) {
    el.status.textContent = msg;
    el.status.classList.toggle("error", !!err);
}

// -- theme --

function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    const next = t === "dark" ? "light" : "dark";
    el.theme.title = `Switch to ${next}`;
    el.theme.setAttribute("aria-label", `Switch to ${next}`);
}
function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    setTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
}
function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    setTheme(next);
}

// -- api --

async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/x-www-form-urlencoded" } };
    if (body) opts.body = new URLSearchParams(body).toString();
    const res = await fetch(path, opts);
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return json;
}

// -- render --

function renderTokens(tokens) {
    el.tokensStage.textContent = "";
    let prev = -1;
    for (const t of tokens) {
        const span = document.createElement("span");
        span.className = "token";
        let c = (t.id * 5) % N_COLORS;
        if (c === prev) c = (c + 1) % N_COLORS;
        prev = c;
        span.style.setProperty("--tk", `var(--tok-${c})`);

        const idEl = document.createElement("span");
        idEl.className = "id";
        idEl.textContent = t.id;
        span.append(idEl, document.createTextNode(esc(t.decoded)));
        span.title = `vocab: ${t.text}`;
        el.tokensStage.appendChild(span);
    }
}

function esc(s) {
    if (s == null) return "";
    return s.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r")
            .replace(/\t/g, "\\t").replace(/\f/g, "\\f")
            .replace(/[\x00-\x08\x0b\x0e-\x1f\x7f]/g, (c) =>
                "\\x" + c.charCodeAt(0).toString(16).padStart(2, "0"));
}

function updateStats(n, chars) {
    el.tokenCount.textContent = n;
    el.charCount.textContent = chars;
    el.ratio.textContent = (n / Math.max(1, chars)).toFixed(2);
    el.statsPills.hidden = false;
}

function clearView() {
    el.tokensStage.textContent = "";
    el.tokenIds.value = "";
    el.tokensCard.hidden = true;
    el.statsPills.hidden = true;
}

// -- actions --

async function loadTokenizers() {
    try {
        const list = await api("GET", "/api/tokenizers");
        el.tokenizer.textContent = "";
        for (const t of list) {
            const o = document.createElement("option");
            o.value = t.id;
            o.textContent = `${t.name} — ${t.description}`;
            el.tokenizer.appendChild(o);
        }
    } catch (e) { status(e.message, true); }
}

async function tokenize() {
    const text = el.text.value;
    if (!text) { clearView(); return status("Enter some text to tokenize.", true); }
    try {
        status("Tokenizing...");
        const r = await api("POST", "/api/tokenize", { tokenizer: el.tokenizer.value, text });
        updateStats(r.tokenCount, r.characters);
        renderTokens(r.tokens);
        el.tokensCard.hidden = false;
        syncIds();
        el.tokenIds.value = r.tokens.map((t) => t.id).join(", ");
        status("Ready");
    } catch (e) { status(e.message, true); }
}

async function decode() {
    const tokens = el.tokenIds.value.trim();
    if (!tokens) return status("Enter token IDs to decode.", true);
    try {
        status("Decoding...");
        const r = await api("POST", "/api/decode", { tokenizer: el.tokenizer.value, tokens });
        el.decoded.textContent = r.text;
        status("Ready");
    } catch (e) { status(e.message, true); }
}

function schedule() {
    clearTimeout(timer);
    timer = setTimeout(tokenize, 350);
}

function syncIds() { el.tokensStage.classList.toggle("hide-ids", !el.showIds.checked); }

// -- events --

el.theme.addEventListener("click", toggleTheme);
el.clear.addEventListener("click", () => { el.text.value = ""; clearView(); el.decoded.textContent = ""; status("Cleared"); el.text.focus(); });
el.decodeBtn.addEventListener("click", decode);
el.copyTokens.addEventListener("click", () => {
    if (!el.tokenIds.value) return;
    navigator.clipboard.writeText(el.tokenIds.value).then(() => { status("Copied"); setTimeout(() => status("Ready"), 1200); });
});
el.text.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") tokenize(); });
el.text.addEventListener("input", schedule);
el.tokenizer.addEventListener("change", tokenize);
el.showIds.addEventListener("change", syncIds);

// -- init --

el.text.value = "The quick brown fox jumps over the lazy dog";
initTheme();
loadTokenizers();
syncIds();
setTimeout(tokenize, 120);
