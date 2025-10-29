async function predict() {
  const apiBase = document.getElementById("apiBase").value;
  const nums = document.getElementById("numbers").value.trim();
  const status = document.getElementById("msg");
  const table = document.getElementById("resultTable");
  const body = document.getElementById("resultBody");

  if (!nums) {
    status.textContent = "⚠️ Lütfen veri gir";
    return;
  }

  // Girdiyi 3 değer olarak alıyoruz (metrekare, oda sayısı, bina yaşı)
  // Örnek giriş: 120,3,10
  const parts = nums.split(",").map((v) => Number(v.trim()));
  if (parts.length !== 3 || parts.some(isNaN)) {
    alert("3 değer gir: metrekare, oda sayısı, bina yaşı (örnek: 120,3,10)");
    return;
  }

  status.textContent = "⏳ Tahmin yapılıyor...";

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: [[parts[0], parts[1], parts[2]]] }),
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    table.style.display = "table";
    body.innerHTML = `
      <tr>
        <td>1</td>
        <td>${parts.join(", ")}</td>
        <td>${data.prediction[0].toFixed(0)} ₺</td>
      </tr>
    `;

    document.getElementById("xrt").textContent =
      res.headers.get("X-Response-Time") || "—";
    document.getElementById("count").textContent = data.count || 1;
    status.textContent = "✅ Tahmin tamamlandı";
  } catch (err) {
    console.error(err);
    status.textContent = "❌ Hata: " + err.message;
  }
}
