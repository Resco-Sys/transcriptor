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
    supportsContext: true,
    contextFlag: "--condition-on-previous-text",
  },
  windows: {
    command: ["whisper-ctranslate2"],
    model: "large-v3-turbo",
    supportsContext: true,
    contextFlag: "--condition_on_previous_text",
    args: ["--device", "cpu", "--compute_type", "int8"],
  },
};

export function engineFor(os) {
  return ENGINES[os] ?? ENGINES.windows;
}

export function buildUvxArgs({ os, audioPath, language, format, useContext = true }) {
  const engine = engineFor(os);
  const args = [...engine.command, audioPath, "--model", engine.model];
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

export function isStatusLine(line) {
  return (
    /^args:/i.test(line) ||
    /detecting language/i.test(line) ||
    /detected language/i.test(line) ||
    /download/i.test(line) ||
    /transcription results written/i.test(line) ||
    /it\/s\]/.test(line) ||
    /\d+%\|/.test(line) ||
    /^error:/i.test(line) ||
    /^traceback/i.test(line) ||
    /^skipping /i.test(line)
  );
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
