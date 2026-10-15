import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  buildUvArgs,
  detectOs,
  engineFor,
  extractSegmentText,
  fileName,
  isPreparationLine,
  isStatusLine,
  joinLinesAsProse,
  joinLinesIntoParagraphs,
  outputFilePath,
  shouldFlattenLines,
  transcriptionError,
} from "./transcript.js";

const AUDIO_EXTENSIONS = [
  "mp3",
  "wav",
  "m4a",
  "flac",
  "ogg",
  "opus",
  "aac",
  "wma",
  "aiff",
  "caf",
  "mp4",
  "mkv",
  "webm",
  "mov",
];

const pickButton = document.querySelector("#pick-file");
const transcribeButton = document.querySelector("#transcribe");
const fileLabel = document.querySelector("#file-label");
const languageSelect = document.querySelector("#language");
const formatSelect = document.querySelector("#format");
const flattenOption = document.querySelector("#flatten-option");
const flattenInput = document.querySelector("#flatten-lines");
const contextOption = document.querySelector("#context-option");
const contextInput = document.querySelector("#use-context");
const controls = document.querySelector("#controls");
const progressEl = document.querySelector("#progress");
const resultEl = document.querySelector("#result");
const statusEl = document.querySelector("#status");

const CONTEXT_KEY = "transcriptor.useContext";
const FLATTEN_KEY = "transcriptor.flattenLines";
const FIRST_RUN_MESSAGE =
  "Preparando por primera vez: descargando componentes y modelo. Puede tardar varios minutos.";

let audioPath = "";
let running = false;
const engine = engineFor(currentOs());

if (engine.supportsContext) {
  const stored = localStorage.getItem(CONTEXT_KEY);
  contextInput.checked = stored === null ? true : stored === "true";
  contextOption.hidden = false;
  contextInput.addEventListener("change", () => {
    localStorage.setItem(CONTEXT_KEY, String(contextInput.checked));
  });
}

const storedFlatten = localStorage.getItem(FLATTEN_KEY);
flattenInput.checked = storedFlatten === null ? true : storedFlatten === "true";
flattenInput.addEventListener("change", () => {
  localStorage.setItem(FLATTEN_KEY, String(flattenInput.checked));
});
formatSelect.addEventListener("change", syncFlatten);
syncFlatten();

function syncFlatten() {
  const unavailable = formatSelect.value !== "txt";
  flattenInput.disabled = unavailable;
  flattenOption.classList.toggle("is-disabled", unavailable);
}

function currentOs() {
  const platform =
    navigator.userAgentData?.platform ||
    navigator.platform ||
    navigator.userAgent;
  return detectOs(platform);
}

function setStatus(message) {
  statusEl.textContent = message;
}

function showPreparing() {
  statusEl.classList.add("preparing");
  setStatus(FIRST_RUN_MESSAGE);
  showPlaceholder(FIRST_RUN_MESSAGE);
}

function setBusy(busy) {
  running = busy;
  controls.disabled = busy;
  transcribeButton.disabled = busy || !audioPath;
}

function showPlaceholder(message) {
  resultEl.replaceChildren();
  const paragraph = document.createElement("p");
  paragraph.className = "placeholder";
  paragraph.textContent = message;
  resultEl.append(paragraph);
}

function showParagraphs(paragraphs) {
  resultEl.replaceChildren();
  if (paragraphs.length === 0) {
    showPlaceholder("La transcripción terminó sin texto reconocible.");
    return;
  }
  for (const text of paragraphs) {
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    resultEl.append(paragraph);
  }
}

function isProgressBar(line) {
  return /\d+%\|/.test(line) || /\d+%\s*\|/.test(line);
}

pickButton.addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    directory: false,
    title: "Elegir archivo de audio",
    filters: [{ name: "Audio", extensions: AUDIO_EXTENSIONS }],
  });

  if (typeof selected !== "string" || !selected) return;

  audioPath = selected;
  fileLabel.textContent = fileName(selected);
  fileLabel.title = selected;
  transcribeButton.disabled = running;
});

transcribeButton.addEventListener("click", async () => {
  if (!audioPath || running) return;

  const os = currentOs();
  const active = engineFor(os);
  const language = languageSelect.value;
  const format = formatSelect.value;
  const flatten = shouldFlattenLines(format, flattenInput.checked);
  const args = buildUvArgs({
    os,
    audioPath,
    language,
    format,
    useContext: active.supportsContext && contextInput.checked,
  });

  const segments = [];
  const prose = [];
  const errors = [];
  const log = [];

  const renderLog = () => {
    progressEl.textContent = log.join("\n");
    progressEl.scrollTop = progressEl.scrollHeight;
  };

  const pushLog = (line) => {
    const clean = line.replace(/\r/g, "").trim();
    if (!clean) return;
    if (isProgressBar(clean) && log.length > 0 && isProgressBar(log[log.length - 1])) {
      log[log.length - 1] = clean;
    } else {
      log.push(clean);
    }
    renderLog();
  };

  const consume = (chunk, collectProse) => {
    for (const piece of String(chunk).split(/\n/)) {
      const line = piece.replace(/\r/g, "").trim();
      if (!line) continue;
      pushLog(line);
      const segment = extractSegmentText(line);
      if (segment) {
        segments.push(segment);
        continue;
      }
      if (
        line.startsWith("Error:") ||
        line.startsWith("Exception Message:") ||
        line.startsWith("RuntimeError:")
      ) {
        errors.push(line);
      }
      if (isPreparationLine(line)) showFirstRun();
      if (collectProse && !isStatusLine(line) && !isProgressBar(line)) {
        prose.push(line);
      }
    }
  };

  let settled = false;
  let revealed = false;
  let stopListening = () => {};

  const showFirstRun = () => {
    if (settled || revealed) return;
    showPreparing();
  };

  const finish = (status, placeholder) => {
    if (settled) return;
    settled = true;
    stopListening();
    statusEl.classList.remove("preparing");
    if (placeholder) showPlaceholder(placeholder);
    setStatus(status);
    setBusy(false);
  };

  const reveal = () => {
    revealed = true;
    const source = segments.length > 0 ? segments : prose;
    const paragraphs = flatten
      ? joinLinesAsProse(source)
      : joinLinesIntoParagraphs(source);
    const failure = transcriptionError(errors);
    if (paragraphs.length > 0) {
      showParagraphs(paragraphs);
    } else if (failure) {
      showPlaceholder(failure);
    } else {
      showParagraphs([]);
    }
  };

  progressEl.textContent = "";
  showPlaceholder("Transcribiendo…");
  statusEl.classList.remove("preparing");
  setStatus("Transcribiendo…");
  setBusy(true);
  pushLog(
    `uv ${args.map((arg) => (/\s/.test(arg) ? `"${arg}"` : arg)).join(" ")}`,
  );

  try {
    const unlistenOutput = await listen("transcription-output", (event) => {
      const payload = event.payload ?? {};
      consume(payload.line ?? "", payload.stream === "stdout");
    });
    stopListening = () => {
      unlistenOutput();
    };
    const unlistenExit = await listen("transcription-exit", async (event) => {
      const code = event.payload?.code;
      let saved = true;
      if (code === 0 && flatten) {
        try {
          await invoke("flatten_transcript", {
            path: outputFilePath(audioPath, "txt"),
          });
        } catch (error) {
          saved = false;
          pushLog(`No se pudo unir el TXT: ${errorText(error)}`);
        }
      }
      reveal();
      finish(
        code === 0
          ? saved
            ? "Transcripción lista."
            : "Transcripción lista. No se pudo unir el TXT."
          : `uv terminó con código ${code ?? "desconocido"}.`,
      );
    });
    stopListening = () => {
      unlistenOutput();
      unlistenExit();
    };

    const result = await invoke("start_transcription", {
      args,
      toolPackage: active.toolPackage,
      toolBin: active.toolBin,
      modelRepo: active.cacheRepo,
    });
    if (result?.firstRun) showFirstRun();
  } catch (error) {
    pushLog(errorText(error));
    finish("No se pudo iniciar uv.", "No se pudo iniciar la transcripción.");
  }
});

function errorText(error) {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
