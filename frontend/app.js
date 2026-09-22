/* JEVLite frontend — set USE_API to true when FastAPI exposes POST /classify. */
const USE_API = true;
const API_URL = "/classify";
const mockScores = { Fattura: 0.87, Contratto: 0.09, "Report interno": 0.03, Altro: 0.01 };
let categories = Object.keys(mockScores);
let uploadedFile = null;

const $ = (selector) => document.querySelector(selector);
const textArea = $("#documentText");
const categoryList = $("#categoryList");

function renderCategories() {
  categoryList.innerHTML = categories.map((category, index) => `<span class="category-chip">${escapeHtml(category)}<button aria-label="Rimuovi ${escapeHtml(category)}" data-index="${index}">×</button></span>`).join("");
  categoryList.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
    categories.splice(Number(button.dataset.index), 1); renderCategories();
  }));
}

function escapeHtml(value) { return value.replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }
function updateCounts() { const chars = textArea.value.length; $("#characterCount").textContent = `${chars.toLocaleString("it-IT")} / 20.000`; $("#wordCount").textContent = `${textArea.value.trim() ? textArea.value.trim().split(/\s+/).length : 0} parole`; }
function selectSource(source) { document.querySelectorAll(".source-tab").forEach(tab => { const active = tab.dataset.source === source; tab.classList.toggle("active", active); tab.setAttribute("aria-selected", active); }); $("#textSource").classList.toggle("hidden", source !== "text"); $("#uploadSource").classList.toggle("hidden", source !== "upload"); }
function showState(state) { $("#emptyState").classList.toggle("hidden", state !== "empty"); $("#loadingState").classList.toggle("hidden", state !== "loading"); $("#resultContent").classList.toggle("hidden", state !== "result"); }

function normalizeResponse(data) {
  // Supports { category, confidence, probabilities, inference_time_ms } from FastAPI.
  const probabilities = data.probabilities || data.scores || {};
  const category = data.category || data.label || Object.keys(probabilities).sort((a,b) => probabilities[b] - probabilities[a])[0];
  const confidence = Number(data.confidence ?? probabilities[category] ?? 0);
  return { category, confidence: confidence > 1 ? confidence / 100 : confidence, probabilities, time: data.inference_time_ms ?? data.latency_ms ?? 248 };
}

function renderResult(result) {
  const scores = Object.entries(result.probabilities).sort((a,b) => b[1] - a[1]);
  const percent = Math.round(result.confidence * 100);
  $("#mainCategory").textContent = result.category;
  $("#mainConfidence").textContent = `${percent}%`;
  $("#confidenceCopy").textContent = percent >= 80 ? "Elevata confidenza nella classificazione" : "Classificazione da verificare";
  $("#confidenceRing").style.setProperty("--progress", 0);
  $("#confidenceList").innerHTML = scores.map(([name, score], i) => { const value = Math.round((Number(score) > 1 ? Number(score)/100 : Number(score))*100); return `<div class="confidence-item ${i === 0 ? "top" : ""}"><div class="confidence-label"><span>${escapeHtml(name)}</span><b>${value}%</b></div><div class="bar-track"><div class="bar-fill" style="width:0" data-width="${value}%"></div></div></div>`; }).join("");
  $("#inferenceTime").textContent = `${Math.round(result.time)} ms`;
  showState("result");
  requestAnimationFrame(() => { $("#confidenceRing").style.setProperty("--progress", percent); document.querySelectorAll(".bar-fill").forEach(bar => bar.style.width = bar.dataset.width); });
}

async function classify() {
  const documentText = textArea.value.trim();
  if (!documentText && !uploadedFile) { textArea.focus(); textArea.placeholder = "Inserisci del testo prima di classificare…"; return; }
  if (!categories.length) { alert("Aggiungi almeno una categoria."); return; }
  const button = $("#classifyButton"); button.classList.add("loading"); button.querySelector("span").textContent = "Analisi in corso…"; showState("loading");
  try {
    let result;
    if (USE_API) {
      const response = await fetch(API_URL, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ document: documentText, categories }) });
      if (!response.ok) throw new Error(`API error ${response.status}`);
      result = normalizeResponse(await response.json()); $("#resultMode").textContent = "LIVE";
    } else {
      await new Promise(resolve => setTimeout(resolve, 900));
      const demo = Object.fromEntries(categories.map((c, i) => [c, mockScores[c] ?? Math.max(.01, .05 - i * .01)]));
      result = normalizeResponse({ category: categories[0], confidence: demo[categories[0]], probabilities: demo, inference_time_ms: 248 }); $("#resultMode").textContent = "DEMO";
    }
    renderResult(result);
  } catch (error) { showState("empty"); console.error(error); alert("Non è stato possibile contattare il classificatore locale. Verifica che FastAPI sia in esecuzione."); }
  finally { button.classList.remove("loading"); button.querySelector("span").textContent = "Classifica documento"; }
}

$("#addCategory").addEventListener("click", () => { const input = document.createElement("input"); input.className = "category-chip category-editor"; input.placeholder = "Nuova categoria"; categoryList.append(input); input.focus(); const save = () => { const value = input.value.trim(); if (value && !categories.includes(value)) categories.push(value); renderCategories(); }; input.addEventListener("keydown", e => { if (e.key === "Enter") save(); if (e.key === "Escape") renderCategories(); }); input.addEventListener("blur", save); });
document.querySelectorAll(".source-tab").forEach(tab => tab.addEventListener("click", () => selectSource(tab.dataset.source)));
textArea.addEventListener("input", updateCounts); $("#classifyButton").addEventListener("click", classify);
const fileInput = $("#fileInput");
function setFile(file) { if (!file) return; uploadedFile = file; $("#fileName").textContent = file.name; $("#fileType").textContent = file.name.split(".").pop().toUpperCase(); $("#fileCard").classList.remove("hidden"); }
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("#removeFile").addEventListener("click", () => { uploadedFile = null; fileInput.value = ""; $("#fileCard").classList.add("hidden"); });
const uploadZone = $("#uploadZone"); ["dragenter","dragover"].forEach(event => uploadZone.addEventListener(event, e => { e.preventDefault(); uploadZone.classList.add("dragging"); })); ["dragleave","drop"].forEach(event => uploadZone.addEventListener(event, e => { e.preventDefault(); uploadZone.classList.remove("dragging"); })); uploadZone.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
renderCategories(); updateCounts();
