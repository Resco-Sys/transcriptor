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
    /resolving \d+ packages/i.test(line)
  );
}

export function isStatusLine(line) {
  return (
    isPreparationLine(line) ||
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

export function shouldFlattenLines(format, checked) {
  return format === "txt" && checked;
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
