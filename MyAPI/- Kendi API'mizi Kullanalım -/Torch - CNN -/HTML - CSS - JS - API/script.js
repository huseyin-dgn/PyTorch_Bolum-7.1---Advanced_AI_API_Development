const els = {
  apiBase: document.getElementById("apiBase"),
  topk: document.getElementById("topk"),
  drop: document.getElementById("drop"),
  file: document.getElementById("file"),
  btnPick: document.getElementById("btnPick"),
  btnPredict: document.getElementById("btnPredict"),
  btnClear: document.getElementById("btnClear"),
  btnHealth: document.getElementById("btnHealth"),
  preview: document.getElementById("preview"),
  status: document.getElementById("status"),
  msg: document.getElementById("msg"),
  predList: document.getElementById("predList"),
  emptyHint: document.getElementById("emptyHint"),
  xrt: document.getElementById("xrt"),
  mdl: document.getElementById("mdl"),
};

let SELECTED_FILE = null;
let LABELS = null;

function setStatus(text, ok = false) {
  els.msg.textContent = text;
  els.status.classList.toggle("ok", ok);
}

function toPercent(p) {
  const v = Math.max(0, Math.min(1, Number(p) || 0));
  return (v * 100).toFixed(1) + "%";
}

function renderPreds(items) {
  els.predList.innerHTML = "";
  items.forEach((it, idx) => {
    const li = document.createElement("li");
    li.className = "pred-item";
    li.innerHTML = `
      <div class="pred-rank">${idx + 1}</div>
      <div class="pred-label">${it.label}</div>
      <div class="pred-prob">${toPercent(it.prob)}</div>
      <div class="bar-wrap"><div class="bar" style="width:0%"></div></div>
    `;
    els.predList.appendChild(li);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        li.querySelector(".bar").style.width = toPercent(it.prob);
      });
    });
  });
  els.predList.hidden = items.length === 0;
  els.emptyHint.style.display = items.length ? "none" : "block";
}

["dragenter", "dragover"].forEach((ev) => {
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.drop.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((ev) => {
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.drop.classList.remove("drag");
  });
});
els.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) onFilePicked(f);
});
els.file.addEventListener("change", () => {
  if (els.file.files && els.file.files[0]) onFilePicked(els.file.files[0]);
});
function onFilePicked(f) {
  SELECTED_FILE = f;
  const url = URL.createObjectURL(f);
  els.preview.src = url;
  els.preview.style.display = "block";
  setStatus(`Görsel yüklendi: ${f.name}`, true);
}

els.btnPick.addEventListener("click", () => els.file.click());
els.btnClear.addEventListener("click", () => {
  els.file.value = "";
  SELECTED_FILE = null;
  els.preview.removeAttribute("src");
  els.preview.style.display = "none";
  renderPreds([]);
  els.xrt.textContent = "—";
  els.mdl.textContent = "—";
  setStatus("Temizlendi");
});
els.btnHealth.addEventListener("click", async () => {
  try {
    const base = els.apiBase.value.trim();
    const res = await fetch(`${base}/health`);
    const data = await res.json();
    setStatus(
      data.ok ? `API hazır (${data.device})` : "API hazır değil",
      data.ok
    );
  } catch {
    setStatus("Sağlık isteği başarısız");
  }
});
els.btnPredict.addEventListener("click", predict);

async function loadLabels() {
  try {
    const base = els.apiBase.value.trim();
    const res = await fetch(`${base}/labels`);
    if (!res.ok) return;
    const data = await res.json();
    LABELS = data.labels || data || null;
  } catch {
    LABELS = null;
  }
}

async function predict() {
  if (!SELECTED_FILE) {
    setStatus("⚠️ Görsel seç", false);
    return;
  }
  els.btnPredict.disabled = true;
  setStatus("⏳ Tahmin yapılıyor...");
  try {
    const base = els.apiBase.value.trim();
    const k = Number(els.topk.value) || 3;
    const fd = new FormData();
    fd.append("file", SELECTED_FILE);
    fd.append("topk", String(k));
    const res = await fetch(`${base}/predict-image`, {
      method: "POST",
      body: fd,
    });
    const xrt = res.headers.get("X-Response-Time") || "—";
    const data = await res.json();
    let preds = [];
    if (Array.isArray(data?.predictions)) preds = data.predictions;
    else if (Array.isArray(data?.topk))
      preds = data.topk.map(([l, p]) => ({ label: l, prob: p }));
    else if (Array.isArray(data?.indices)) {
      const labels = LABELS || data.labels || [];
      preds = data.indices.map((i, ix) => ({
        label: labels[i] ?? `class_${i}`,
        prob: data.probs[ix],
      }));
    }
    renderPreds(preds);
    els.xrt.textContent = xrt;
    els.mdl.textContent = data.model || "—";
    setStatus("✅ Tahmin tamamlandı", true);
  } catch (err) {
    console.error(err);
    setStatus("❌ Hata: tahmin alınamadı");
  } finally {
    els.btnPredict.disabled = false;
  }
}

(async function init() {
  await loadLabels();
})();
