// Cliente: dispara la prediccion y pinta el resultado.
const $ = (id) => document.getElementById(id);
const COLS = ["fecha", "local", "visita", "goles_local", "goles_visita", "xg_local",
  "xg_visita", "posesion_local", "tiros_local", "tiros_visita", "felyne_cluster", "ganador", "origen"];
const HEADERS = { felyne_cluster: "CLUSTER" };  // etiqueta visible != nombre interno
const head = (c) => HEADERS[c] || c.replace(/_/g, " ").toUpperCase();

$("run").addEventListener("click", async () => {
  const local = $("local").value, visita = $("visita").value, k = $("k").value;
  if (local === visita) { $("status").textContent = "ERROR: SELECCIONA DOS EQUIPOS DISTINTOS."; return; }

  $("run").disabled = true;
  $("status").textContent = "ENTRENANDO K-MEANS Y ARBOL ID3...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ local, visita, k })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "FALLO EN EL MODELO");
    paintVerdict(data); paintAnalysis(data); paintData(data);
    $("status").textContent = `PREDICCION COMPLETADA ${data.timestamp} — ${data.muestras} PARTIDOS PROCESADOS.`;
    $("f-samples").textContent = `MUESTRAS: ${data.muestras}`;
  } catch (e) {
    $("status").textContent = "ERROR: " + e.message.toUpperCase();
  } finally { $("run").disabled = false; }
});

$("export").addEventListener("click", () => {
  window.location = `/api/export?local=${encodeURIComponent($("local").value)}&visita=${encodeURIComponent($("visita").value)}`;
});

function paintVerdict(d) {
  $("verdict").classList.remove("is-hidden");
  $("v-winner").textContent = d.ganador;
  $("v-score").textContent = `${d.local.slice(0, 3)} ${d.marcador} ${d.visita.slice(0, 3)}`;
  const top = Math.max(...Object.values(d.probabilidades));
  $("v-conf").textContent = top.toFixed(1) + "%";

  const nombres = { LOCAL: d.local, EMPATE: "EMPATE", VISITANTE: d.visita };
  $("v-probs").innerHTML = Object.entries(d.probabilidades).map(([k, v]) => `
    <div class="bar-row ${k === d.veredicto ? "win" : ""}">
      <span class="name">${nombres[k].slice(0, 10)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${v}%"></span></span>
      <span>${v.toFixed(1)}%</span>
    </div>`).join("");

  $("v-scorers").innerHTML = d.anotadores.length
    ? d.anotadores.map(a => `<li>${a.jugador} <small>(${a.equipo.slice(0, 3)}/${a.posicion})</small><b>${a.prob}%</b></li>`).join("")
    : "<li>PARTIDO SIN GOLES ESPERADOS</li>";

  $("v-alt").textContent = "XG ESTIMADO " + d.xg_estimado.local + " - " + d.xg_estimado.visita +
    " // MARCADORES ALTERNATIVOS: " + d.marcadores_alternativos.map(m => `${m.marcador} (${m.prob}%)`).join("  ·  ");
}

function paintAnalysis(d) {
  $("analysis").classList.remove("is-hidden");
  $("a-cluster").textContent = `CLUSTER ${d.cluster_activo} — ${d.cluster_nombre}`;

  const keys = Object.keys(d.centroides[0]).filter(k => k !== "nombre");
  $("a-centroids").innerHTML =
    `<tr>${keys.map(k => `<th>${head(k)}</th>`).join("")}</tr>` +
    d.centroides.map(c => `<tr class="${c.cluster === d.cluster_activo ? "active" : ""}">
      ${keys.map(k => `<td>${c[k]}</td>`).join("")}</tr>`).join("");

  $("a-path").innerHTML = d.ruta_id3.map(p => `<li>${p}</li>`).join("");
  $("a-rules").textContent = d.reglas_id3;
}

function paintData(d) {
  $("data").classList.remove("is-hidden");
  $("d-table").innerHTML =
    `<tr>${COLS.map(c => `<th>${head(c)}</th>`).join("")}</tr>` +
    d.tabla.map(r => `<tr>${COLS.map(c =>
      `<td class="${c === "felyne_cluster" ? "c" : ""}">${r[c]}</td>`).join("")}</tr>`).join("");
}
