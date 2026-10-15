use std::env;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

static RUNNING: AtomicBool = AtomicBool::new(false);

struct ClearRunning;

impl Drop for ClearRunning {
    fn drop(&mut self) {
        RUNNING.store(false, Ordering::SeqCst);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartResponse {
    first_run: bool,
}

#[derive(Clone, Serialize)]
struct OutputEvent {
    stream: &'static str,
    line: String,
}

#[derive(Clone, Serialize)]
struct ExitEvent {
    code: Option<i32>,
}

#[tauri::command]
pub fn start_transcription(
    app: AppHandle,
    args: Vec<String>,
    tool_package: String,
    tool_bin: String,
    model_repo: String,
) -> Result<StartResponse, String> {
    if args.is_empty() {
        return Err("Faltan los argumentos de transcripción.".into());
    }
    validate_name(&tool_package)?;
    validate_name(&tool_bin)?;
    if !model_repo.contains('/') || model_repo.contains("..") || model_repo.contains('\\') {
        return Err("Repositorio del modelo no válido.".into());
    }

    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err("Ya hay una transcripción en curso.".into());
    }
    let guard = ClearRunning;

    let first_run = needs_first_run(&tool_package, &tool_bin, &model_repo);
    let app_dir = app_binary_dir()?;
    let uv = find_sidecar(&app_dir, "uv")?;
    let ffmpeg = find_sidecar(&app_dir, "ffmpeg")?;
    let ffmpeg_dir = ffmpeg_directory(&app, &ffmpeg)?;
    let path_value = prepend_path(&ffmpeg_dir)?;

    let mut command = Command::new(&uv);
    command
        .args(&args)
        .env(path_key(), path_value)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            drop(guard);
            return Err(format!("No se pudo iniciar {}: {error}", uv.display()));
        }
    };

    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            drop(guard);
            return Err("uv no expuso stdout.".into());
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            drop(guard);
            return Err("uv no expuso stderr.".into());
        }
    };
    let output_app = app.clone();
    let exit_app = app.clone();

    thread::spawn(move || {
        let _guard = guard;
        let stdout_app = output_app.clone();
        let stderr_app = output_app;
        let stdout_thread = thread::spawn(move || emit_output(&stdout_app, "stdout", stdout));
        let stderr_thread = thread::spawn(move || emit_output(&stderr_app, "stderr", stderr));
        let status = child.wait();
        let _ = stdout_thread.join();
        let _ = stderr_thread.join();
        let code = status.ok().and_then(|status| status.code());
        let _ = exit_app.emit("transcription-exit", ExitEvent { code });
    });

    Ok(StartResponse { first_run })
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains('\0')
    {
        return Err("Nombre de herramienta no válido.".into());
    }
    Ok(())
}

/// Carpeta del ejecutable. Tauri copia ahí los sidecars y les quita el target triple,
/// así que `uv-x86_64-pc-windows-msvc.exe` queda como `uv.exe`.
fn app_binary_dir() -> Result<PathBuf, String> {
    let exe =
        env::current_exe().map_err(|error| format!("No se pudo ubicar la aplicación: {error}"))?;
    let mut dir = exe
        .parent()
        .ok_or("La aplicación no tiene carpeta.")?
        .to_path_buf();
    if dir.file_name().and_then(|name| name.to_str()) == Some("deps") {
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        }
    }
    Ok(dir)
}

fn binary_file_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn suffixed_file_name(name: &str) -> String {
    let triple = env!("TAURI_ENV_TARGET_TRIPLE");
    if cfg!(windows) {
        format!("{name}-{triple}.exe")
    } else {
        format!("{name}-{triple}")
    }
}

fn is_usable_binary(path: &Path) -> bool {
    fs::metadata(path)
        .map(|meta| meta.is_file() && meta.len() > 0)
        .unwrap_or(false)
}

fn find_sidecar(app_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let plain = binary_file_name(name);
    let suffixed = suffixed_file_name(name);
    for dir in [app_dir.to_path_buf(), app_dir.join("binaries")] {
        for file_name in [&plain, &suffixed] {
            let candidate = dir.join(file_name);
            if is_usable_binary(&candidate) {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "No se encontró el sidecar {plain} junto a la aplicación. Ejecuta la app con tauri dev o tauri build para empaquetarlo."
    ))
}

/// El proceso de Whisper busca `ffmpeg` en el PATH. Tauri normalmente ya lo
/// deja con ese nombre al quitar el target triple. Si el archivo sigue
/// llevando el triple, se publica una copia llamada `ffmpeg` (o `ffmpeg.exe`).
fn ffmpeg_directory(app: &AppHandle, ffmpeg: &Path) -> Result<PathBuf, String> {
    let wanted = binary_file_name("ffmpeg");
    let file_name = ffmpeg
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let parent = ffmpeg
        .parent()
        .ok_or("ffmpeg no tiene carpeta.")?
        .to_path_buf();

    if file_name == wanted {
        return Ok(parent);
    }

    let beside = parent.join(&wanted);
    if is_usable_binary(&beside) || link_or_copy(ffmpeg, &beside).is_ok() {
        return Ok(parent);
    }

    let fallback = app
        .path()
        .app_local_data_dir()
        .map_err(|error| format!("No se pudo crear una carpeta para ffmpeg: {error}"))?
        .join("sidecars");
    let dest = fallback.join(&wanted);
    if !is_usable_binary(&dest) {
        link_or_copy(ffmpeg, &dest).map_err(|error| {
            format!("No se pudo exponer ffmpeg con el nombre {wanted}: {error}")
        })?;
    }
    Ok(fallback)
}

fn link_or_copy(src: &Path, dest: &Path) -> std::io::Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::hard_link(src, dest).or_else(|_| {
        fs::copy(src, dest)?;
        Ok(())
    })
}

fn path_key() -> std::ffi::OsString {
    env::vars_os()
        .find_map(|(key, _)| {
            key.to_string_lossy()
                .eq_ignore_ascii_case("path")
                .then_some(key)
        })
        .unwrap_or_else(|| "PATH".into())
}

fn prepend_path(dir: &Path) -> Result<std::ffi::OsString, String> {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(current) = env::var_os("PATH") {
        paths.extend(env::split_paths(&current));
    }
    env::join_paths(paths).map_err(|error| format!("No se pudo preparar PATH: {error}"))
}

fn needs_first_run(package: &str, bin: &str, repo: &str) -> bool {
    !model_cached(repo) || !tool_cached(package, bin)
}

fn model_cached(repo: &str) -> bool {
    let Some(hub) = huggingface_hub_dir() else {
        return false;
    };
    let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let revision = dir.join("refs").join("main");
    if fs::metadata(&revision)
        .map(|meta| meta.is_file() && meta.len() > 0)
        .unwrap_or(false)
    {
        return true;
    }
    contains_large_file(&dir.join("snapshots"), 50_000_000, 4, 4_000)
}

fn tool_cached(package: &str, bin: &str) -> bool {
    let executable = binary_file_name(bin);
    if let Some(tools) = uv_tools_dir() {
        if tree_has_file(&tools.join(package), &executable, 4, 4_000) {
            return true;
        }
    }
    let Some(cache) = uv_cache_dir() else {
        return false;
    };
    tree_has_file(&cache.join("environments-v2"), &executable, 6, 20_000)
        || archive_has_dist(&cache.join("archive-v0"), package)
}

fn huggingface_hub_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("HUGGINGFACE_HUB_CACHE").or_else(|| env::var_os("HF_HUB_CACHE"))
    {
        return Some(PathBuf::from(dir));
    }
    if let Some(dir) = env::var_os("HF_HOME") {
        return Some(PathBuf::from(dir).join("hub"));
    }
    home_dir().map(|home| home.join(".cache").join("huggingface").join("hub"))
}

fn uv_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("UV_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    {
        env::var_os("LOCALAPPDATA").map(|dir| PathBuf::from(dir).join("uv").join("cache"))
    }
    #[cfg(not(windows))]
    {
        if let Some(dir) = env::var_os("XDG_CACHE_HOME") {
            return Some(PathBuf::from(dir).join("uv"));
        }
        home_dir().map(|home| home.join(".cache").join("uv"))
    }
}

fn uv_tools_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("UV_TOOL_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    {
        env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("uv").join("data").join("tools"))
    }
    #[cfg(not(windows))]
    {
        let data = if let Some(dir) = env::var_os("XDG_DATA_HOME") {
            PathBuf::from(dir).join("uv")
        } else {
            home_dir()?.join(".local").join("share").join("uv")
        };
        Some(data.join("tools"))
    }
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        env::var_os("HOME").map(PathBuf::from)
    }
}

fn tree_has_file(root: &Path, file_name: &str, max_depth: u32, max_entries: u32) -> bool {
    if !root.is_dir() {
        return false;
    }
    let mut stack = vec![(root.to_path_buf(), 0u32)];
    let mut seen = 0u32;
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            seen += 1;
            if seen > max_entries {
                return false;
            }
            let path = entry.path();
            if entry.file_name() == file_name && path.is_file() {
                return true;
            }
            if depth < max_depth && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    false
}

fn contains_large_file(root: &Path, min_bytes: u64, max_depth: u32, max_entries: u32) -> bool {
    if !root.is_dir() {
        return false;
    }
    let mut stack = vec![(root.to_path_buf(), 0u32)];
    let mut seen = 0u32;
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            seen += 1;
            if seen > max_entries {
                return false;
            }
            let path = entry.path();
            if path.is_file()
                && fs::metadata(&path)
                    .map(|meta| meta.len() >= min_bytes)
                    .unwrap_or(false)
            {
                return true;
            }
            if depth < max_depth && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    false
}

fn archive_has_dist(archive: &Path, package: &str) -> bool {
    let prefix = format!("{}-", package.replace('-', "_"));
    let Ok(entries) = fs::read_dir(archive) else {
        return false;
    };
    for (index, entry) in entries.flatten().enumerate() {
        if index >= 8_000 {
            break;
        }
        let Ok(children) = fs::read_dir(entry.path()) else {
            continue;
        };
        for child in children.flatten() {
            let name = child.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".dist-info") {
                return true;
            }
        }
    }
    false
}

fn emit_output(app: &AppHandle, stream: &'static str, reader: impl Read) {
    let mut reader = BufReader::new(reader);
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut pending = String::new();

    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => {
                bytes.extend_from_slice(&chunk[..count]);
                let valid = match std::str::from_utf8(&bytes) {
                    Ok(text) => text.len(),
                    Err(error) => error.valid_up_to(),
                };
                if valid == 0 {
                    continue;
                }
                pending.push_str(&String::from_utf8_lossy(&bytes[..valid]));
                bytes.drain(..valid);
                for line in take_lines(&mut pending) {
                    emit_line(app, stream, line);
                }
            }
            Err(_) => break,
        }
    }

    if !bytes.is_empty() {
        pending.push_str(&String::from_utf8_lossy(&bytes));
    }
    for line in take_lines(&mut pending) {
        emit_line(app, stream, line);
    }
    if !pending.is_empty() {
        emit_line(app, stream, std::mem::take(&mut pending));
    }
}

fn take_lines(pending: &mut String) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        let bytes = pending.as_bytes();
        let Some(index) = bytes
            .iter()
            .position(|byte| *byte == b'\n' || *byte == b'\r')
        else {
            break;
        };
        let mut next = index + 1;
        if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            next = index + 2;
        }
        let line = pending[..index].to_string();
        pending.drain(..next);
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

fn emit_line(app: &AppHandle, stream: &'static str, line: String) {
    let _ = app.emit("transcription-output", OutputEvent { stream, line });
}
