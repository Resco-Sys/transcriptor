const SEGMENT_RE =
  /^\[(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}\s-->\s(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}\]\s*(.*)$/;

const SENTENCE_END = /[.!?…]["»”')\]]*$/;

export function parentDirectory(path) {
  const index = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  if (index < 0) return ".";
  if (index === 2 && path[1] === ":") return path.slice(0, index + 1);
  if (index === 0) return path.slice(0, 1);
  return path.slice(0, index);
}

export function fileName(path) {
  const index = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return index < 0 ? path : path.slice(index + 1);
}

export function detectOs(platform = "") {
  if (/mac/i.test(platform) && !/win/i.test(platform)) return "macos";
  if (/win/i.test(platform)) return "windows";
  return "other";
}

export const ENGINES = {
  macos: {
    command: ["--from", "mlx-whisper", "mlx_whisper"],
    model: "mlx-community/whisper-large-v3-turbo",
    cacheRepo: "mlx-community/whisper-large-v3-turbo",
    toolPackage: "mlx-whisper",
    toolBin: "mlx_whisper",
    supportsContext: true,
    contextFlag: "--condition-on-previous-text",
  },
  windows: {
    command: ["whisper-ctranslate2"],
    model: "large-v3-turbo",
    cacheRepo: "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    toolPackage: "whisper-ctranslate2",
    toolBin: "whisper-ctranslate2",
    supportsContext: true,
    contextFlag: "--condition_on_previous_text",
    args: ["--device", "cpu", "--compute_type", "int8"],
  },
};

export function engineFor(os) {
  return ENGINES[os] ?? ENGINES.windows;
}

export function buildUvArgs({ os, audioPath, language, format, useContext = true }) {
  const engine = engineFor(os);
  const args = ["tool", "run", ...engine.command, audioPath, "--model", engine.model];
  if (language) args.push("--language", language);
  args.push("-f", format, "-o", parentDirectory(audioPath), "--verbose", "True");
  if (engine.args) args.push(...engine.args);
  if (engine.supportsContext) {
    args.push(engine.contextFlag, useContext ? "True" : "False");
  }
  return args;
}

export function transcriptionError(lines) {
  const exception = lines.find((line) => line.startsWith("Exception Message:"));
  if (exception) return exception.slice("Exception Message:".length).trim();
  const runtime = lines.find((line) => line.startsWith("RuntimeError:"));
  if (runtime) return runtime.slice("RuntimeError:".length).trim();
  const error = lines.find((line) => line.startsWith("Error:"));
  if (error) return error.slice("Error:".length).trim();
  return "";
}

export function extractSegmentText(line) {
  const match = line.match(SEGMENT_RE);
  if (!match) return null;
  const text = match[1].replace(/\s+/g, " ").trim();
  return text || null;
}

export function isPreparationLine(line) {
  return (
    /download/i.test(line) ||
    /\bfetch(?:ing)?\b/i.test(line) ||
    /installed \d+ packages/i.test(line) ||
    /cpython-/i.test(line) ||
    /resolving \d+ packages/i.test(line) ||
    /resolved \d+ packages/i.test(line)
  );
}

export const DIARIZE_STAGES = [
  "Transcribiendo…",
  "Identificando hablantes…",
  "Combinando…",
];

export const DIARIZE_MODEL_URL =
  "https://huggingface.co/pyannote/speaker-diarization-community-1";

export const DIARIZE_HINT_MISSING =
  "Para distinguir hablantes, agrega un token en Configuración.";

export const DIARIZE_HINT_UNTESTED =
  "Prueba el token en Configuración para distinguir hablantes.";

export function isDiarizeStage(line) {
  return DIARIZE_STAGES.includes(String(line).trim());
}

export function isStatusLine(line) {
  return (
    isPreparationLine(line) ||
    isDiarizeStage(line) ||
    String(line).startsWith("PARAGRAPH\t") ||
    /^dispositivo:/i.test(line) ||
    /^args:/i.test(line) ||
    /detecting language/i.test(line) ||
    /detected language/i.test(line) ||
    /transcription results written/i.test(line) ||
    /it\/s\]/.test(line) ||
    /\d+%\|/.test(line) ||
    /^error:/i.test(line) ||
    /^traceback/i.test(line) ||
    /^skipping /i.test(line)
  );
}

export function outputFilePath(audioPath, extension) {
  const index = Math.max(audioPath.lastIndexOf("/"), audioPath.lastIndexOf("\\"));
  const directory = index < 0 ? "" : audioPath.slice(0, index + 1);
  const base = index < 0 ? audioPath : audioPath.slice(index + 1);
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  return `${directory}${stem}.${extension}`;
}

export function shouldFlattenLines(format, checked, diarize = false) {
  return format === "txt" && checked && !diarize;
}

export function joinLinesAsProse(lines) {
  const text = lines
    .map((line) => String(line).replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  return text ? [text] : [];
}

export function joinLinesIntoParagraphs(lines) {
  const paragraphs = [];
  let parts = [];
  let sentences = 0;

  const flush = () => {
    if (parts.length === 0) return;
    paragraphs.push(parts.join(" "));
    parts = [];
    sentences = 0;
  };

  for (const raw of lines) {
    const line = String(raw).replace(/\s+/g, " ").trim();
    if (!line) {
      flush();
      continue;
    }
    parts.push(line);
    if (SENTENCE_END.test(line)) sentences += 1;
    const joined = parts.join(" ");
    if ((sentences >= 3 && SENTENCE_END.test(line)) || joined.length >= 480) {
      flush();
    }
  }

  flush();
  return paragraphs;
}

export function buildDiarizeArgs({
  os = "macos",
  audioPath,
  language,
  format,
  useContext = true,
  numSpeakers = "",
}) {
  const args = [
    "run",
    "--no-project",
    "diarize.py",
    "--audio",
    audioPath,
    "--model",
    diarizeModel(os),
    "--format",
    format,
    "--output-dir",
    parentDirectory(audioPath),
    "--condition-on-previous-text",
    useContext ? "true" : "false",
  ];
  if (language) args.push("--language", language);
  const count = speakerCount(numSpeakers);
  if (count) args.push("--num-speakers", count);
  return args;
}

export function speakerCount(value) {
  const count = String(value);
  return count === "2" || count === "3" || count === "4" ? count : "";
}

export function diarizeSupported(os) {
  return os === "macos" || os === "windows";
}

export function diarizeModel(os) {
  return os === "windows" ? ENGINES.windows.model : ENGINES.macos.model;
}

export function tokenStoreHelp(os) {
  if (os === "windows") {
    return "Se guarda en el Administrador de credenciales de Windows. No vuelve a mostrarse.";
  }
  return "Se guarda en el Llavero de macOS. No vuelve a mostrarse.";
}

export function tokenSavedMessage(os) {
  if (os === "windows") {
    return "Token guardado en el Administrador de credenciales. Pulsa Probar para comprobar el acceso.";
  }
  return "Token guardado en el Llavero. Pulsa Probar para comprobar el acceso.";
}

export function diarizePanel({ os, saved = false, valid = false }) {
  if (!diarizeSupported(os)) {
    return { settings: false, option: false, checkbox: false, hint: "" };
  }
  if (saved && valid) {
    return { settings: true, option: true, checkbox: true, hint: "" };
  }
  return {
    settings: true,
    option: true,
    checkbox: false,
    hint: saved ? DIARIZE_HINT_UNTESTED : DIARIZE_HINT_MISSING,
  };
}

export function tokenTestMessage(code) {
  switch (code) {
    case "ok":
      return {
        tone: "ok",
        text: "El token es válido y tiene acceso al modelo.",
        link: false,
      };
    case "missing":
      return {
        tone: "error",
        text: "No hay un token guardado. Escríbelo y pulsa Guardar.",
        link: false,
      };
    case "invalid":
      return { tone: "error", text: "El token no es válido.", link: false };
    case "gated":
      return {
        tone: "error",
        text: "El token es válido, pero falta aceptar las condiciones del modelo.",
        link: true,
      };
    case "unsaved":
      return {
        tone: "error",
        text: "Pulsa Guardar antes de probar el token.",
        link: false,
      };
    default:
      return {
        tone: "error",
        text: "No se pudo contactar con Hugging Face. Revisa la conexión.",
        link: false,
      };
  }
}

export function maskSecrets(line) {
  return String(line)
    .replace(/hf_[A-Za-z0-9]{8,}/g, "hf_***")
    .replace(/(authorization:\s*bearer\s+)\S+/gi, "$1***")
    .replace(/([?&]token=)[^&\s]+/gi, "$1***");
}

export function parseSpeakerLine(line) {
  const match = /^(Persona \d+):\s*(.*)$/.exec(String(line).trim());
  if (!match) return null;
  return { speaker: match[1], text: match[2] };
}

export function liveSpeakerName(value) {
  return String(value).replace(/[\r\n:]/g, "").slice(0, 60);
}

export function sanitizeSpeakerName(value, fallback) {
  const clean = String(value)
    .replace(/[\r\n:]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 60);
  return clean || fallback;
}
