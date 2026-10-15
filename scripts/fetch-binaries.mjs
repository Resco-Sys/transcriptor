#!/usr/bin/env node
// Descarga uv y ffmpeg para la plataforma actual y los deja en
// src-tauri/binaries con el target triple que exige Tauri.
// Si el archivo de destino ya existe, no descarga nada.
//
// uv: release fijo de https://github.com/astral-sh/uv (no "latest").
//
// ffmpeg estático, también con versión fija:
// - Windows y Linux: BtbN/FFmpeg-Builds, el proveedor de binarios de Windows
//   enlazado desde ffmpeg.org. El archivo es el build GPL estático de
//   FFmpeg 9.0.2 (sin "-shared": los códecs van dentro del ejecutable),
//   del release autobuild-2026-10-15-13-03. No se usa el tag móvil "latest".
// - macOS: builds estáticos de Martin Riedl (ffmpeg.martin-riedl.de),
//   FFmpeg 9.0.2, firmados, para Apple Silicon e Intel. ffmpeg.org apunta
//   a evermeet.cx, pero ese proyecto solo publica binarios Intel y dice
//   que no va a ofrecer Apple Silicon.

import { execFileSync } from "node:child_process";
import { createWriteStream } from "node:fs";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const UV_VERSION = "0.13.1";
const BTB_N_TAG = "autobuild-2026-10-15-13-03";
const BTB_N_BUILD = "n9.0.2-28-g669f5ab083";

const FFMPEG_URLS = {
  "x86_64-pc-windows-msvc": btbN("win64-gpl-9.0.zip"),
  "aarch64-pc-windows-msvc": btbN("winarm64-gpl-9.0.zip"),
  "x86_64-unknown-linux-gnu": btbN("linux64-gpl-9.0.tar.xz"),
  "aarch64-unknown-linux-gnu": btbN("linuxarm64-gpl-9.0.tar.xz"),
  "aarch64-apple-darwin":
    "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffmpeg.zip",
  "x86_64-apple-darwin":
    "https://ffmpeg.martin-riedl.de/download/macos/amd64/1789931006_9.0.2/ffmpeg.zip",
};

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const binariesDir = path.join(root, "src-tauri", "binaries");
const MIN_BYTES = 1024 * 1024;

function btbN(asset) {
  return `https://github.com/BtbN/FFmpeg-Builds/releases/download/${BTB_N_TAG}/ffmpeg-${BTB_N_BUILD}-${asset}`;
}

function targetTriple() {
  try {
    const printed = execFileSync("rustc", ["--print", "host-tuple"], {
      encoding: "utf8",
    }).trim();
    if (printed && !printed.includes(" ")) return printed;
  } catch {
    // rustc anterior a 1.84 no tiene host-tuple.
  }

  const verbose = execFileSync("rustc", ["-Vv"], { encoding: "utf8" });
  const match = verbose.match(/^host:\s+(\S+)/m);
  if (!match) {
    throw new Error("No se pudo leer el target triple de rustc.");
  }
  return match[1];
}

function uvUrl(triple) {
  const extension = triple.includes("windows") ? "zip" : "tar.gz";
  return `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${triple}.${extension}`;
}

function destination(name, triple) {
  const extension = triple.includes("windows") ? ".exe" : "";
  return path.join(binariesDir, `${name}-${triple}${extension}`);
}

function alreadyFetched(file) {
  return fs.existsSync(file) && fs.statSync(file).size >= MIN_BYTES;
}

async function download(url, file) {
  const response = await fetch(url, {
    redirect: "follow",
    headers: { "user-agent": "transcriptor-fetch-binaries" },
  });
  if (!response.ok || !response.body) {
    throw new Error(`No se pudo descargar ${url} (HTTP ${response.status}).`);
  }
  await pipeline(Readable.fromWeb(response.body), createWriteStream(file));
}

function findExecutable(dir, baseName) {
  const wanted = new Set([baseName, `${baseName}.exe`]);
  const stack = [dir];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        continue;
      }
      if (wanted.has(entry.name)) return full;
    }
  }
  return null;
}

function markExecutable(file) {
  if (process.platform === "win32") return;
  fs.chmodSync(file, 0o755);
  if (process.platform === "darwin") {
    try {
      execFileSync("xattr", ["-d", "com.apple.quarantine", file], {
        stdio: "ignore",
      });
    } catch {
      // El atributo no siempre está presente.
    }
  }
}

async function fetchBinary({ name, url, triple }) {
  const dest = destination(name, triple);
  if (alreadyFetched(dest)) {
    console.log(`${name} ya está en ${path.relative(root, dest)}`);
    return;
  }

  fs.mkdirSync(binariesDir, { recursive: true });
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), `transcriptor-${name}-`));
  const archive = path.join(tempRoot, path.basename(new URL(url).pathname));
  const extracted = path.join(tempRoot, "out");
  try {
    console.log(`Descargando ${name}: ${url}`);
    await download(url, archive);
    fs.mkdirSync(extracted);
    execFileSync("tar", ["-xf", archive, "-C", extracted], { stdio: "inherit" });
    const found = findExecutable(extracted, name);
    if (!found || found === archive) {
      throw new Error(`El archivo de ${name} no contiene el ejecutable.`);
    }
    if (fs.statSync(found).size < MIN_BYTES) {
      throw new Error(`El ejecutable de ${name} descargado está incompleto.`);
    }
    const partial = `${dest}.partial`;
    fs.copyFileSync(found, partial);
    markExecutable(partial);
    if (fs.existsSync(dest)) fs.rmSync(dest);
    fs.renameSync(partial, dest);
    console.log(`Listo: ${path.relative(root, dest)}`);
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
}

const triple = targetTriple();
const ffmpegUrl = FFMPEG_URLS[triple];
if (!ffmpegUrl) {
  console.error(`No hay un build de ffmpeg configurado para ${triple}.`);
  process.exit(1);
}

await fetchBinary({ name: "uv", url: uvUrl(triple), triple });
await fetchBinary({ name: "ffmpeg", url: ffmpegUrl, triple });
