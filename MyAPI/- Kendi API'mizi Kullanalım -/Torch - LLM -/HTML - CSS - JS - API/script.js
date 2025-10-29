// ========== Yardımcılar ==========
const $ = (s) => document.querySelector(s);
const chat = $("#chat");
const promptEl = $("#prompt");
const apiBaseEl = $("#apiBase");
const latEl = $("#lat");
const xrtEl = $("#xrt");
const tokEl = $("#tok");
const toast = $("#toast");
const healthDot = $("#healthDot");
const btnSend = $("#btnSend");
const btnReload = $("#btnReload");
const btnDocs = $("#btnDocs");
const btnClear = $("#btnClear");
const btnCopyLast = $("#btnCopyLast");
const btnExport = $("#btnExport");
const tempEl = $("#temperature");
const tempVal = $("#temperatureVal");
const maxNewEl = $("#maxNew");
const maxNewVal = $("#maxNewVal");
const modelInfo = $("#modelInfo");

let lastAI = "";

// ========== Persist ==========
apiBaseEl.value = localStorage.getItem("apiBase") || apiBaseEl.value;
apiBaseEl.addEventListener("change", () => {
  localStorage.setItem("apiBase", apiBaseEl.value.trim());
  ping();
});

// ========== UI ==========
function toastShow(msg, ms = 1800) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), ms);
}
function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `
    <div class="role">${role === "user" ? "👤" : "🤖"}</div>
    <div class="bubble">${escapeHTML(text).replace(/\n/g, "<br>")}</div>
  `;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}
function escapeHTML(s) {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[
        c
      ])
  );
}

// sliders
function syncSliders() {
  tempVal.textContent = Number(tempEl.value).toFixed(2);
  maxNewVal.textContent = maxNewEl.value;
}
tempEl.addEventListener("input", syncSliders);
maxNewEl.addEventListener("input", syncSliders);
syncSliders();

// ========== Health ==========
async function ping() {
  try {
    const t0 = performance.now();
    const res = await fetch(`${apiBaseEl.value}/health`);
    const t1 = performance.now();
    const data = await res.json();
    healthDot.classList.toggle("ok", !!data.ok);
    healthDot.classList.toggle("bad", !data.ok);
    modelInfo.textContent = `${data.ok ? "hazır" : "hazır değil"} • ${
      data.device || "?"
    } • ${Math.round(t1 - t0)}ms`;
  } catch {
    healthDot.classList.remove("ok");
    healthDot.classList.add("bad");
    modelInfo.textContent = "erişilemiyor";
  }
}
ping();
setInterval(ping, 5000);

// ========== Gönder ==========
async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    toastShow("Önce bir prompt yaz 🙂");
    promptEl.focus();
    return;
  }

  addMsg("user", prompt);

  const payload = {
    prompt,
    temperature: Number(tempEl.value),
    max_new_tokens: Number(maxNewEl.value),
  };

  const t0 = performance.now();
  let xResponse = "—";

  try {
    const res = await fetch(`${apiBaseEl.value}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    xResponse = res.headers.get("X-Response-Time") || "—";
    const data = await res.json();

    if (!res.ok || data.error)
      throw new Error(data.error || `HTTP ${res.status}`);

    const t1 = performance.now();
    latEl.textContent = `${(t1 - t0).toFixed(0)}ms`;
    xrtEl.textContent = xResponse;
    tokEl.textContent = Array.isArray(data.tokens) ? data.tokens.length : "—";

    lastAI = data.response || "";
    addMsg("ai", lastAI || "—");
  } catch (err) {
    addMsg("ai", "❌ Hata: " + err.message);
    toastShow("İstek başarısız: " + err.message, 2200);
  }
}

// ========== Reload ==========
async function reloadModel() {
  try {
    const res = await fetch(`${apiBaseEl.value}/reload`, { method: "POST" });
    const data = await res.json();
    if (!res.ok || data.error)
      throw new Error(data.error || `HTTP ${res.status}`);
    toastShow("Model yeniden yüklendi ✓");
    ping();
  } catch (err) {
    toastShow("Yükleme hatası: " + err.message, 2200);
  }
}

// ========== Kısayollar & Butonlar ==========
btnSend.addEventListener("click", generate);
btnReload.addEventListener("click", reloadModel);
btnClear.addEventListener("click", () => {
  chat.innerHTML = "";
  lastAI = "";
  toastShow("Konuşma temizlendi");
});
btnDocs.addEventListener("click", () => {
  addMsg(
    "sys",
    [
      "<b>Kullanım İpuçları</b>",
      "• Eğitimde kullandığın prompt formatını (ör. <i>Soru:\nCevap:</i>) aynı şekilde gönder.",
      "• <b>Temperature</b> düşükse (0) deterministik, yüksekse daha yaratıcı.",
      "• <b>Max Tokens</b> üretilecek yeni token sayısını sınırlar.",
    ].join("<br>")
  );
});
btnCopyLast.addEventListener("click", async () => {
  if (!lastAI) return toastShow("Henüz cevap yok");
  await navigator.clipboard.writeText(lastAI);
  toastShow("Kopyalandı ✓");
});
btnExport.addEventListener("click", () => {
  const lines = [...document.querySelectorAll(".msg")]
    .map((el) => {
      const role = el.classList.contains("user")
        ? "USER"
        : el.classList.contains("ai")
        ? "AI"
        : "SYS";
      const text = el.querySelector(".bubble")?.innerText || "";
      return `# ${role}\n${text}\n`;
    })
    .join("\n");
  const blob = new Blob([lines], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `chat-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    generate();
  }
});
