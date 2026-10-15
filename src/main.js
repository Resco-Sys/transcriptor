import { open } from "@tauri-apps/plugin-dialog";
import { Command } from "@tauri-apps/plugin-shell";
import {
  buildUvxArgs,
  detectOs,
  extractSegmentText,
  fileName,
  isStatusLine,
  joinLinesIntoParagraphs,
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
const controls = document.querySelector("#controls");
const progressEl = document.querySelector("#progress");
const resultEl = document.querySelector("#result");
const statusEl = document.querySelector("#status");

let audioPath = "";
let running = false;

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

  const language = languageSelect.value;
  const format = formatSelect.value;
  const args = buildUvxArgs({
    os: currentOs(),
    audioPath,
    language,
    format,
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
      if (collectProse && !isStatusLine(line) && !isProgressBar(line)) {
        prose.push(line);
      }
    }
  };

  let settled = false;
  const finish = (status, placeholder) => {
    if (settled) return;
    settled = true;
    if (placeholder) showPlaceholder(placeholder);
    setStatus(status);
    setBusy(false);
  };

  progressEl.textContent = "";
  showPlaceholder("Transcribiendo…");
  setStatus("Transcribiendo…");
  setBusy(true);
  pushLog(
    `uvx ${args.map((arg) => (/\s/.test(arg) ? `"${arg}"` : arg)).join(" ")}`,
  );

  const command = Command.create("uvx", args, {
    env: {
      PYTHONIOENCODING: "utf-8",
      PYTHONUTF8: "1",
    },
  });

  command.stdout.on("data", (line) => consume(line, true));
  command.stderr.on("data", (line) => consume(line, false));
  command.on("error", (error) => {
    pushLog(errorText(error));
  });

  command.on("close", (data) => {
    const source = segments.length > 0 ? segments : prose;
    const paragraphs = joinLinesIntoParagraphs(source);
    const failure = transcriptionError(errors);
    if (paragraphs.length > 0) {
      showParagraphs(paragraphs);
    } else if (failure) {
      showPlaceholder(failure);
    } else {
      showParagraphs([]);
    }
    finish(
      data.code === 0
        ? "Transcripción lista."
        : `uvx terminó con código ${data.code ?? "desconocido"}.`,
    );
  });

  try {
    await command.spawn();
  } catch (error) {
    pushLog(errorText(error));
    finish("No se pudo iniciar uvx.", "No se pudo iniciar la transcripción.");
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
