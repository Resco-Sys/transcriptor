import assert from "node:assert/strict";
import test from "node:test";
import {
  DIARIZE_HINT_MISSING,
  DIARIZE_HINT_UNTESTED,
  DIARIZE_MODEL_URL,
  buildDiarizeArgs,
  buildUvArgs,
  diarizePanel,
  tokenSavedMessage,
  tokenStoreHelp,
  isDiarizeStage,
  isPreparationLine,
  isStatusLine,
  maskSecrets,
  parseSpeakerLine,
  sanitizeSpeakerName,
  shouldFlattenLines,
  tokenTestMessage,
} from "./transcript.js";

test("diarization is available on macOS and Windows", () => {
  assert.equal(diarizePanel({ os: "windows", saved: true, valid: true }).checkbox, true);
  assert.equal(diarizePanel({ os: "linux", saved: true, valid: true }).settings, false);
  assert.match(tokenStoreHelp("windows"), /Administrador de credenciales/);
  assert.match(tokenSavedMessage("windows"), /Administrador de credenciales/);
  assert.match(tokenStoreHelp("macos"), /Llavero/);
});

test("macOS explains where to add a token", () => {
  const missing = diarizePanel({ os: "macos", saved: false, valid: false });
  assert.equal(missing.checkbox, false);
  assert.equal(missing.settings, true);
  assert.equal(missing.hint, DIARIZE_HINT_MISSING);

  const untested = diarizePanel({ os: "macos", saved: true, valid: false });
  assert.equal(untested.checkbox, false);
  assert.equal(untested.hint, DIARIZE_HINT_UNTESTED);

  const ready = diarizePanel({ os: "macos", saved: true, valid: true });
  assert.equal(ready.checkbox, true);
  assert.equal(ready.hint, "");
});

test("diarize command never carries the token", () => {
  const args = buildDiarizeArgs({
    audioPath: "/tmp/entrevista.wav",
    language: "es",
    format: "srt",
    useContext: false,
    numSpeakers: "2",
  });
  assert.equal(args.includes("diarize.py"), true);
  assert.equal(args.includes("--num-speakers"), true);
  assert.equal(args.at(-1), "2");
  assert.match(args.join(" "), /--condition-on-previous-text false/);
  assert.equal(args.join(" ").includes("HF_TOKEN"), false);
  assert.equal(args.join(" ").includes("hf_"), false);

  const automatic = buildDiarizeArgs({
    audioPath: "/tmp/entrevista.wav",
    language: "",
    format: "txt",
    useContext: true,
    numSpeakers: "",
  });
  assert.equal(automatic.includes("--num-speakers"), false);
  assert.equal(automatic.includes("--language"), false);
  assert.equal(args.includes("mlx-community/whisper-large-v3-turbo"), true);

  const windows = buildDiarizeArgs({
    os: "windows",
    audioPath: "C:\\entrevista.wav",
    language: "es",
    format: "txt",
    useContext: true,
    numSpeakers: "3",
  });
  assert.equal(windows.includes("large-v3-turbo"), true);
  assert.equal(windows.includes("mlx-community/whisper-large-v3-turbo"), false);
  assert.equal(windows.includes("HF_TOKEN"), false);

  const normal = buildUvArgs({
    os: "windows",
    audioPath: "C:\\audio.mp3",
    language: "es",
    format: "txt",
  });
  assert.equal(normal.includes("diarize.py"), false);
  assert.deepEqual(normal.slice(0, 5), [
    "tool",
    "run",
    "--python",
    "3.12",
    "whisper-ctranslate2",
  ]);

  const macos = buildUvArgs({
    os: "macos",
    audioPath: "/tmp/audio.mp3",
    language: "es",
    format: "txt",
  });
  assert.deepEqual(macos.slice(0, 7), [
    "tool",
    "run",
    "--python",
    "3.12",
    "--from",
    "mlx-whisper",
    "mlx_whisper",
  ]);
});

test("speaker turns stay separate from flattened prose", () => {
  assert.equal(shouldFlattenLines("txt", true, false), true);
  assert.equal(shouldFlattenLines("txt", true, true), false);
  assert.equal(shouldFlattenLines("srt", true, false), false);
});

test("stage lines are progress, not transcript text", () => {
  assert.equal(isDiarizeStage("Identificando hablantes…"), true);
  assert.equal(isStatusLine("Combinando…"), true);
  assert.equal(isStatusLine("PARAGRAPH\tPersona 1: Hola"), true);
  assert.equal(isPreparationLine("Resolved 40 packages in 2s"), true);
  assert.equal(isPreparationLine("Downloading torch (174.2MiB)"), true);
});

test("secrets are masked and gated access links to the model", () => {
  const line = "usando hf_abcdefghijklmnop y Authorization: Bearer hf_abcdefghijklmnop ?token=hf_abcdefghijklmnop";
  const masked = maskSecrets(line);
  assert.equal(masked.includes("abcdefghijklmnop"), false);
  assert.match(masked, /hf_\*\*\*/);
  const gated = tokenTestMessage("gated");
  assert.equal(gated.link, true);
  assert.match(gated.text, /condiciones/);
  assert.match(DIARIZE_MODEL_URL, /speaker-diarization-community-1$/);
});

test("speaker lines keep the label and the spoken text apart", () => {
  assert.deepEqual(parseSpeakerLine("Persona 2: Hola, Ana"), {
    speaker: "Persona 2",
    text: "Hola, Ana",
  });
  assert.equal(parseSpeakerLine("sin etiqueta"), null);
  assert.equal(sanitizeSpeakerName("  Ana: María \n", "Persona 1"), "Ana María");
  assert.equal(sanitizeSpeakerName("   ", "Persona 1"), "Persona 1");
});
