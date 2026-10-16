import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  DIARIZE_MODEL_URL,
  buildDiarizeArgs,
  buildUvArgs,
  detectOs,
  diarizePanel,
  diarizeSupported,
  tokenSavedMessage,
  tokenStoreHelp,
  engineFor,
  extractSegmentText,
  fileName,
  isDiarizeStage,
  isPreparationLine,
  isStatusLine,
  joinLinesAsProse,
  joinLinesIntoParagraphs,
  liveSpeakerName,
  maskSecrets,
  outputFilePath,
  parseSpeakerLine,
  sanitizeSpeakerName,
  shouldFlattenLines,
  speakerCount,
  tokenTestMessage,
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

const FLATTEN_HELP =
  "El TXT se guarda y se muestra como un solo párrafo. Con SRT no se aplica.";
const DIARIZE_FLATTEN_HELP =
  "Con hablantes, el texto se guarda en párrafos por persona.";

const pickButton = document.querySelector("#pick-file");
const transcribeButton = document.querySelector("#transcribe");
const fileLabel = document.querySelector("#file-label");
const languageSelect = document.querySelector("#language");
const formatSelect = document.querySelector("#format");
const flattenOption = document.querySelector("#flatten-option");
const flattenInput = document.querySelector("#flatten-lines");
const flattenHelp = document.querySelector("#flatten-help");
const contextOption = document.querySelector("#context-option");
const contextInput = document.querySelector("#use-context");
const controls = document.querySelector("#controls");
const progressEl = document.querySelector("#progress");
const resultEl = document.querySelector("#result");
const statusEl = document.querySelector("#status");
const settingsEl = document.querySelector("#settings");
const tokenInput = document.querySelector("#hf-token");
const saveTokenButton = document.querySelector("#save-token");
const testTokenButton = document.querySelector("#test-token");
const tokenMessage = document.querySelector("#token-message");
const tokenStoreHelpEl = document.querySelector("#token-store-help");
const diarizeOption = document.querySelector("#diarize-option");
const diarizeCheck = document.querySelector("#diarize-check");
const diarizeInput = document.querySelector("#diarize");
const diarizeHint = document.querySelector("#diarize-hint");
const speakersField = document.querySelector("#speakers-field");
const numSpeakersSelect = document.querySelector("#num-speakers");
const speakerNamesEl = document.querySelector("#speaker-names");

const CONTEXT_KEY = "transcriptor.useContext";
const FLATTEN_KEY = "transcriptor.flattenLines";
const FIRST_RUN_MESSAGE =
  "Preparando por primera vez: descargando componentes y modelo. Puede tardar varios minutos.";

let audioPath = "";
let running = false;
let speakerEdit = null;
let speakerGeneration = 0;
let renameTimer = 0;
let saveQueue = Promise.resolve();
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
diarizeInput.addEventListener("change", () => {
  syncSpeakers();
  syncFlatten();
});
saveTokenButton.addEventListener("click", () => {
  void saveToken();
});
testTokenButton.addEventListener("click", () => {
  void testToken();
});
tokenInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    void saveToken();
  }
});
syncFlatten();
syncSpeakers();

tokenStoreHelpEl.textContent = tokenStoreHelp(currentOs());
if (diarizeSupported(currentOs())) {
  settingsEl.hidden = false;
  void refreshTokenStatus();
} else {
  applyTokenStatus({ saved: false, valid: false });
}

function syncFlatten() {
  const diarize = diarizeInput.checked && !diarizeCheck.hidden;
  const unavailable = formatSelect.value !== "txt" || diarize;
  flattenInput.disabled = unavailable;
  flattenOption.classList.toggle("is-disabled", unavailable);
  flattenHelp.textContent = diarize ? DIARIZE_FLATTEN_HELP : FLATTEN_HELP;
}

function syncSpeakers() {
  speakersField.hidden = !(diarizeInput.checked && !diarizeCheck.hidden);
}

function currentOs() {
  const platform =
    navigator.userAgentData?.platform ||
    navigator.platform ||
    navigator.userAgent;
  return detectOs(platform);
}

function setStatus(message) {
  statusEl.textContent = maskSecrets(message);
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
  saveTokenButton.disabled = busy;
  testTokenButton.disabled = busy;
  tokenInput.disabled = busy;
}

function showPlaceholder(message) {
  resultEl.replaceChildren();
  const paragraph = document.createElement("p");
  paragraph.className = "placeholder";
  paragraph.textContent = maskSecrets(message);
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

function applyTokenStatus(status) {
  const panel = diarizePanel({
    os: currentOs(),
    saved: Boolean(status?.saved),
    valid: Boolean(status?.valid),
  });
  settingsEl.hidden = !panel.settings;
  diarizeOption.hidden = !panel.option;
  diarizeCheck.hidden = !panel.checkbox;
  diarizeHint.hidden = !panel.hint;
  diarizeHint.textContent = panel.hint;
  tokenInput.placeholder = status?.saved ? "Token guardado" : "Pega el token";
  if (!panel.checkbox) diarizeInput.checked = false;
  syncSpeakers();
  syncFlatten();
}

async function refreshTokenStatus() {
  try {
    const status = await invoke("hf_token_status");
    applyTokenStatus(status);
    return status;
  } catch (error) {
    applyTokenStatus({ saved: false, valid: false });
    setTokenMessage({
      tone: "error",
      text: maskSecrets(errorText(error)),
      link: false,
    });
    return null;
  }
}

async function saveToken() {
  if (running) return;
  const token = tokenInput.value.trim();
  if (!token) {
    setTokenMessage({ tone: "error", text: "Escribe un token.", link: false });
    return;
  }
  saveTokenButton.disabled = true;
  try {
    await invoke("save_hf_token", { token });
    tokenInput.value = "";
    setTokenMessage({
      tone: "ok",
      text: tokenSavedMessage(currentOs()),
      link: false,
    });
    await refreshTokenStatus();
  } catch (error) {
    const safe = errorText(error).split(token).join("hf_***");
    setTokenMessage({ tone: "error", text: maskSecrets(safe), link: false });
  } finally {
    saveTokenButton.disabled = running;
  }
}

async function testToken() {
  if (running) return;
  if (tokenInput.value.trim()) {
    setTokenMessage(tokenTestMessage("unsaved"));
    return;
  }
  testTokenButton.disabled = true;
  setTokenMessage({ tone: "", text: "Comprobando el token…", link: false });
  try {
    const result = await invoke("test_hf_token");
    setTokenMessage(tokenTestMessage(result?.code));
    await refreshTokenStatus();
  } catch (error) {
    setTokenMessage({
      tone: "error",
      text: maskSecrets(errorText(error)),
      link: false,
    });
  } finally {
    testTokenButton.disabled = running;
  }
}

function setTokenMessage(message) {
  tokenMessage.className = `help token-message${message.tone ? ` is-${message.tone}` : ""}`;
  tokenMessage.replaceChildren();
  if (!message.text) return;
  tokenMessage.append(message.text);
  if (!message.link) return;
  tokenMessage.append(" ");
  const link = document.createElement("a");
  link.href = DIARIZE_MODEL_URL;
  link.textContent = "Abrir la página del modelo";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.addEventListener("click", (event) => {
    event.preventDefault();
    void openExternal(DIARIZE_MODEL_URL);
  });
  tokenMessage.append(link);
}

async function openExternal(url) {
  try {
    await invoke("plugin:opener|open_url", { url });
  } catch {
    window.open(url, "_blank", "noopener");
  }
}

function clearSpeakerEditor() {
  speakerGeneration += 1;
  clearTimeout(renameTimer);
  speakerEdit = null;
  speakerNamesEl.hidden = true;
  speakerNamesEl.replaceChildren();
}

function showSpeakerResult(lines, outputPath) {
  const turns = [];
  const speakers = [];
  for (const line of lines) {
    const turn = parseSpeakerLine(line);
    if (!turn) continue;
    if (!speakers.some((speaker) => speaker.id === turn.speaker)) {
      speakers.push({
        id: turn.speaker,
        fileLabel: turn.speaker,
        name: turn.speaker,
      });
    }
    turns.push(turn);
  }
  if (turns.length === 0) {
    showParagraphs(lines);
    return;
  }
  speakerEdit = { path: outputPath, turns, speakers };
  renderSpeakerEditors();
  renderSpeakerParagraphs();
}

function renderSpeakerEditors() {
  speakerNamesEl.replaceChildren();
  const title = document.createElement("p");
  title.className = "speaker-title";
  title.textContent = "Hablantes";
  speakerNamesEl.append(title);
  for (const speaker of speakerEdit.speakers) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = speaker.id;
    const input = document.createElement("input");
    input.type = "text";
    input.value = speaker.name;
    input.maxLength = 60;
    input.autocomplete = "off";
    input.spellcheck = false;
    input.addEventListener("input", () => {
      const live = liveSpeakerName(input.value);
      if (input.value !== live) input.value = live;
      speaker.name = live;
      renderSpeakerParagraphs();
      scheduleSpeakerSave();
    });
    input.addEventListener("blur", () => {
      const name = sanitizeSpeakerName(speaker.name, speaker.id);
      speaker.name = name;
      input.value = name;
      renderSpeakerParagraphs();
      clearTimeout(renameTimer);
      void flushSpeakerSave();
    });
    label.append(caption, input);
    speakerNamesEl.append(label);
  }
  const help = document.createElement("p");
  help.className = "help";
  help.textContent = "El nombre se aplica al texto y al archivo guardado.";
  speakerNamesEl.append(help);
  speakerNamesEl.hidden = false;
}

function renderSpeakerParagraphs() {
  if (!speakerEdit) return;
  resultEl.replaceChildren();
  for (const turn of speakerEdit.turns) {
    const speaker = speakerEdit.speakers.find((item) => item.id === turn.speaker);
    const name = (speaker?.name || "").trim() || speaker?.id || turn.speaker;
    const paragraph = document.createElement("p");
    paragraph.textContent = `${name}: ${turn.text}`;
    resultEl.append(paragraph);
  }
}

function scheduleSpeakerSave() {
  clearTimeout(renameTimer);
  const generation = speakerGeneration;
  renameTimer = setTimeout(() => {
    if (generation !== speakerGeneration) return;
    void flushSpeakerSave();
  }, 400);
}

function flushSpeakerSave() {
  const generation = speakerGeneration;
  const edit = speakerEdit;
  if (!edit) return saveQueue;
  saveQueue = saveQueue.then(async () => {
    if (generation !== speakerGeneration || speakerEdit !== edit) return;
    const renames = [];
    for (const speaker of edit.speakers) {
      const name = sanitizeSpeakerName(speaker.name, speaker.id);
      if (name !== speaker.fileLabel) {
        renames.push({ from: speaker.fileLabel, to: name, speaker });
      }
    }
    if (renames.length === 0) return;
    try {
      await invoke("apply_speaker_names", {
        path: edit.path,
        renames: renames.map(({ from, to }) => ({ from, to })),
      });
      if (generation !== speakerGeneration || speakerEdit !== edit) return;
      for (const rename of renames) rename.speaker.fileLabel = rename.to;
    } catch (error) {
      if (generation !== speakerGeneration) return;
      setStatus(`No se pudieron guardar los nombres. ${errorText(error)}`);
    }
  });
  return saveQueue;
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
  const diarize =
    diarizeSupported(os) && diarizeInput.checked && !diarizeCheck.hidden;
  const flatten = shouldFlattenLines(format, flattenInput.checked, diarize);
  const useContext = active.supportsContext && contextInput.checked;
  const args = diarize
    ? buildDiarizeArgs({
        os,
        audioPath,
        language,
        format,
        useContext,
        numSpeakers: speakerCount(numSpeakersSelect.value),
      })
    : buildUvArgs({ os, audioPath, language, format, useContext });

  const segments = [];
  const prose = [];
  const errors = [];
  const log = [];
  const diarizedLines = [];

  const renderLog = () => {
    progressEl.textContent = log.join("\n");
    progressEl.scrollTop = progressEl.scrollHeight;
  };

  const pushLog = (line) => {
    const clean = maskSecrets(line.replace(/\r/g, "").trim());
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
      if (line.startsWith("PARAGRAPH\t")) {
        const text = maskSecrets(line.slice("PARAGRAPH\t".length).trim());
        if (text) diarizedLines.push(text);
        continue;
      }
      pushLog(line);
      if (isDiarizeStage(line)) {
        statusEl.classList.remove("preparing");
        setStatus(line);
      }
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
      if (!diarize && collectProse && !isStatusLine(line) && !isProgressBar(line)) {
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

  const revealDiarized = () => {
    revealed = true;
    const failure = transcriptionError(errors.map(maskSecrets));
    if (diarizedLines.length > 0) {
      showSpeakerResult(diarizedLines, outputFilePath(audioPath, format));
    } else if (failure) {
      showPlaceholder(failure);
    } else {
      showParagraphs([]);
    }
  };

  clearSpeakerEditor();
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
      if (code === 0 && flatten && !diarize) {
        try {
          await invoke("flatten_transcript", {
            path: outputFilePath(audioPath, "txt"),
          });
        } catch (error) {
          saved = false;
          pushLog(`No se pudo unir el TXT: ${errorText(error)}`);
        }
      }
      if (diarize) revealDiarized();
      else reveal();
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
      diarize,
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
