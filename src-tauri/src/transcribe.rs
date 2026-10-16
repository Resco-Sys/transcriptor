use std::env;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use serde::{Deserialize, Serialize};
use tauri::path::BaseDirectory;
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
    diarize: bool,
) -> Result<StartResponse, String> {
    if args.is_empty() {
        return Err("Faltan los argumentos de transcripción.".into());
    }
    validate_name(&tool_package)?;
    validate_name(&tool_bin)?;
    if !model_repo.contains('/') || model_repo.contains("..") || model_repo.contains('\\') {
        return Err("Repositorio del modelo no válido.".into());
    }
    if diarize {
        if !diarization_supported() {
            return Err("Distinguir hablantes solo está disponible en macOS y Windows.".into());
        }
        if !args.iter().any(|arg| arg == "diarize.py") {
            return Err("Falta el script de diarización.".into());
        }
    }

    if RUNNING.swap(true, Ordering::SeqCst) {
        return Err("Ya hay una transcripción en curso.".into());
    }
    let guard = ClearRunning;

    let hf_token = if diarize {
        match crate::secrets::require_valid_token() {
            Ok(token) => Some(token),
            Err(error) => {
                drop(guard);
                return Err(error);
            }
        }
    } else {
        None
    };

    let first_run = if diarize {
        !diarize_dependencies_ready()
    } else {
        needs_first_run(&tool_package, &tool_bin, &model_repo)
    };
    let app_dir = app_binary_dir()?;
    let uv = find_sidecar(&app_dir, "uv")?;
    let ffmpeg = find_sidecar(&app_dir, "ffmpeg")?;
    let ffmpeg_dir = ffmpeg_directory(&app, &ffmpeg)?;
    let path_value = prepend_path(&ffmpeg_dir)?;
    let args = match prepare_args(&app, args, &ffmpeg, diarize) {
        Ok(args) => args,
        Err(error) => {
            drop(guard);
            return Err(error);
        }
    };
    if let Some(token) = &hf_token {
        if token.len() >= 10 && args.iter().any(|arg| arg.contains(token)) {
            drop(guard);
            return Err("El token no puede ir en la línea de comandos.".into());
        }
    }

    let mut command = Command::new(&uv);
    command
        .args(&args)
        .env(path_key(), path_value)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HF_TOKEN")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(token) = &hf_token {
        command.env("HF_TOKEN", token);
    }

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
    let stdout_token = hf_token.clone();
    let stderr_token = hf_token;

    thread::spawn(move || {
        let _guard = guard;
        let stdout_app = output_app.clone();
        let stderr_app = output_app;
        let stdout_thread = thread::spawn(move || {
            emit_output(&stdout_app, "stdout", stdout, stdout_token.as_deref())
        });
        let stderr_thread = thread::spawn(move || {
            emit_output(&stderr_app, "stderr", stderr, stderr_token.as_deref())
        });
        let status = child.wait();
        let _ = stdout_thread.join();
        let _ = stderr_thread.join();
        let code = status.ok().and_then(|status| status.code());
        let _ = exit_app.emit("transcription-exit", ExitEvent { code });
    });

    Ok(StartResponse { first_run })
}

#[tauri::command]
pub fn flatten_transcript(path: String) -> Result<(), String> {
    let file = PathBuf::from(&path);
    let extension = file
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if !extension.eq_ignore_ascii_case("txt") || path.contains('\0') {
        return Err("Solo se pueden unir archivos TXT.".into());
    }
    if !file.is_file() {
        return Err(format!("No se encontró el TXT: {path}"));
    }

    let text =
        fs::read_to_string(&file).map_err(|error| format!("No se pudo leer el TXT: {error}"))?;
    let flat = flatten_line_breaks(&text);
    fs::write(&file, flat).map_err(|error| format!("No se pudo guardar el TXT: {error}"))
}

#[derive(Deserialize)]
pub struct SpeakerRename {
    from: String,
    to: String,
}

#[tauri::command]
pub fn apply_speaker_names(path: String, renames: Vec<SpeakerRename>) -> Result<(), String> {
    let file = PathBuf::from(&path);
    let extension = file
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if !(extension.eq_ignore_ascii_case("txt") || extension.eq_ignore_ascii_case("srt"))
        || path.contains('\0')
    {
        return Err("Solo se pueden actualizar archivos TXT o SRT.".into());
    }
    if !file.is_file() {
        return Err(format!("No se encontró el archivo: {path}"));
    }

    let cleaned = clean_renames(&renames)?;
    let text = fs::read_to_string(&file)
        .map_err(|error| format!("No se pudo leer el archivo: {error}"))?;
    let updated = apply_renames(&text, &cleaned)?;
    if updated != text {
        fs::write(&file, updated)
            .map_err(|error| format!("No se pudo guardar el archivo: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        apply_renames, dist_info_prefix, flatten_line_breaks, is_dist_info_name, mask_secrets,
    };

    #[test]
    fn joins_segment_lines_into_one_paragraph() {
        let flat = flatten_line_breaks("Una frase larga.\nOtra.\r\n\r\nYa.\n");
        assert_eq!(flat, "Una frase larga. Otra. Ya.\n");
    }

    #[test]
    fn keeps_an_empty_file_empty() {
        assert_eq!(flatten_line_breaks(" \n\n "), "");
    }

    #[test]
    fn masks_tokens_in_logs() {
        let token = "hf_abcdefghijklmnop";
        let masked = mask_secrets(
            &format!("descargando {token} Bearer secretvalue ?token={token}"),
            Some(token),
        );
        assert!(!masked.contains("abcdefghijklmnop"));
        assert!(!masked.contains("secretvalue"));
        assert!(masked.contains("hf_***"));
        assert!(masked.contains("Bearer ***"));
        assert!(masked.contains("token=***"));
    }

    #[test]
    fn renames_speakers_without_touching_longer_labels() {
        let text = "Persona 1: Hola.\n\nPersona 10: Sigo.\n";
        let updated = apply_renames(
            text,
            &[
                ("Persona 1".into(), "Ana".into()),
                ("Persona 10".into(), "Luis".into()),
            ],
        )
        .unwrap();
        assert_eq!(updated, "Ana: Hola.\n\nLuis: Sigo.\n");
    }

    #[test]
    fn swaps_speaker_names_in_srt() {
        let text = "1\n00:00:00,000 --> 00:00:01,000\nPersona 1: Hola\n\n2\n00:00:01,000 --> 00:00:02,000\nPersona 2: Qué tal\n";
        let updated = apply_renames(
            text,
            &[
                ("Persona 1".into(), "Persona 2".into()),
                ("Persona 2".into(), "Persona 1".into()),
            ],
        )
        .unwrap();
        assert!(updated.contains("Persona 2: Hola"));
        assert!(updated.contains("Persona 1: Qué tal"));
        assert!(!updated.contains("Persona 1: Hola"));
    }

    #[test]
    fn dist_info_names_match_the_package_version() {
        assert_eq!(dist_info_prefix("pyannote.audio"), "pyannote_audio-");
        assert_eq!(dist_info_prefix("mlx-whisper"), "mlx_whisper-");
        assert!(is_dist_info_name("torch-2.5.1.dist-info", "torch-"));
        assert!(!is_dist_info_name("torchaudio-2.5.1.dist-info", "torch-"));
        assert!(!is_dist_info_name(
            "torch-geometric-2.0.0.dist-info",
            "torch-"
        ));
    }
}

fn flatten_line_breaks(text: &str) -> String {
    let mut flat = String::with_capacity(text.len());
    let mut pending_space = false;
    let mut started = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if started {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            flat.push(' ');
            pending_space = false;
        }
        flat.push(ch);
        started = true;
    }
    if started {
        flat.push('\n');
    }
    flat
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

fn diarization_supported() -> bool {
    cfg!(target_os = "macos") || cfg!(target_os = "windows")
}

fn diarize_dependencies_ready() -> bool {
    let whisper_ready = if cfg!(target_os = "windows") {
        model_cached("mobiuslabsgmbh/faster-whisper-large-v3-turbo")
            && ["faster-whisper", "ctranslate2"]
                .into_iter()
                .all(package_cached)
    } else {
        model_cached("mlx-community/whisper-large-v3-turbo") && package_cached("mlx-whisper")
    };
    whisper_ready
        && model_cached("pyannote/speaker-diarization-community-1")
        && ["torch", "torchaudio", "pyannote.audio"]
            .into_iter()
            .all(package_cached)
}

fn package_cached(package: &str) -> bool {
    let Some(cache) = uv_cache_dir() else {
        return false;
    };
    let prefix = dist_info_prefix(package);
    archive_has_dist(&cache.join("archive-v0"), package)
        || tree_has_dist_info(&cache.join("environments-v2"), &prefix, 8, 20_000)
}

fn prepare_args(
    app: &AppHandle,
    mut args: Vec<String>,
    ffmpeg: &Path,
    diarize: bool,
) -> Result<Vec<String>, String> {
    if !diarize {
        return Ok(args);
    }
    let index = args
        .iter()
        .position(|arg| arg == "diarize.py")
        .ok_or("Falta el script de diarización.")?;
    args[index] = diarize_script(app)?.to_string_lossy().into_owned();
    if !args.iter().any(|arg| arg == "--ffmpeg") {
        args.push("--ffmpeg".into());
        args.push(ffmpeg.to_string_lossy().into_owned());
    }
    Ok(args)
}

fn diarize_script(app: &AppHandle) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Ok(path) = app
        .path()
        .resolve("resources/diarize.py", BaseDirectory::Resource)
    {
        candidates.push(path);
    }
    if let Ok(path) = app.path().resolve("diarize.py", BaseDirectory::Resource) {
        candidates.push(path);
    }
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join("resources").join("diarize.py"));
        candidates.push(dir.join("diarize.py"));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("diarize.py"),
    );
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            "No se encontró el script de diarización. Vuelve a compilar la aplicación.".into()
        })
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
    let prefix = dist_info_prefix(package);
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
            if is_dist_info_name(&name, &prefix) {
                return true;
            }
        }
    }
    false
}

fn dist_info_prefix(package: &str) -> String {
    let mut normalized = String::new();
    let mut pending_sep = false;
    for ch in package.chars() {
        if ch == '-' || ch == '.' || ch == '_' {
            pending_sep = !normalized.is_empty();
            continue;
        }
        if pending_sep {
            normalized.push('_');
            pending_sep = false;
        }
        normalized.push(ch.to_ascii_lowercase());
    }
    normalized.push('-');
    normalized
}

fn is_dist_info_name(name: &str, prefix: &str) -> bool {
    let Some(rest) = name.strip_prefix(prefix) else {
        return false;
    };
    let Some(stem) = rest.strip_suffix(".dist-info") else {
        return false;
    };
    stem.starts_with(|ch: char| ch.is_ascii_digit())
}

fn tree_has_dist_info(root: &Path, prefix: &str, max_depth: u32, max_entries: u32) -> bool {
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
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() && is_dist_info_name(&name, prefix) {
                return true;
            }
            if depth < max_depth && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    false
}

fn emit_output(app: &AppHandle, stream: &'static str, reader: impl Read, token: Option<&str>) {
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
                    emit_line(app, stream, line, token);
                }
            }
            Err(_) => break,
        }
    }

    if !bytes.is_empty() {
        pending.push_str(&String::from_utf8_lossy(&bytes));
    }
    for line in take_lines(&mut pending) {
        emit_line(app, stream, line, token);
    }
    if !pending.is_empty() {
        emit_line(app, stream, std::mem::take(&mut pending), token);
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

fn emit_line(app: &AppHandle, stream: &'static str, line: String, token: Option<&str>) {
    let line = mask_secrets(&line, token);
    let _ = app.emit("transcription-output", OutputEvent { stream, line });
}

fn mask_secrets(line: &str, token: Option<&str>) -> String {
    let mut masked = line.to_string();
    if let Some(token) = token {
        if token.len() >= 10 {
            masked = masked.replace(token, "hf_***");
        }
    }
    masked = mask_hf_tokens(&masked);
    masked = mask_bearer(&masked);
    mask_query_token(&masked)
}

fn mask_hf_tokens(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"hf_") {
            let mut end = index + 3;
            while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
                end += 1;
            }
            if end - (index + 3) >= 8 {
                out.push_str("hf_***");
                index = end;
                continue;
            }
        }
        let ch = line[index..].chars().next().unwrap();
        out.push(ch);
        index += ch.len_utf8();
    }
    out
}

fn mask_bearer(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut index = 0;
    let needle = b"bearer ";
    while index < bytes.len() {
        if index + needle.len() <= bytes.len() && &bytes[index..index + needle.len()] == needle {
            out.push_str(&line[index..index + needle.len()]);
            index += needle.len();
            let start = index;
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
                index += 1;
            }
            if index > start {
                out.push_str("***");
            }
            continue;
        }
        let ch = line[index..].chars().next().unwrap();
        out.push(ch);
        index += ch.len_utf8();
    }
    out
}

fn mask_query_token(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut index = 0;
    while index < bytes.len() {
        let preceded = index == 0 || matches!(bytes[index - 1], b'?' | b'&' | b' ' | b'"' | b'\'');
        if preceded && bytes[index..].starts_with(b"token=") {
            out.push_str("token=");
            index += "token=".len();
            let start = index;
            while index < bytes.len() && !matches!(bytes[index], b'&' | b' ' | b'\t' | b'"' | b'\'')
            {
                index += 1;
            }
            if index > start {
                out.push_str("***");
            }
            continue;
        }
        let ch = line[index..].chars().next().unwrap();
        out.push(ch);
        index += ch.len_utf8();
    }
    out
}

fn clean_renames(renames: &[SpeakerRename]) -> Result<Vec<(String, String)>, String> {
    if renames.len() > 32 {
        return Err("Hay demasiados hablantes.".into());
    }
    renames
        .iter()
        .map(|rename| {
            let from = rename.from.trim();
            let to = rename.to.trim();
            validate_speaker_label(from)?;
            validate_speaker_label(to)?;
            Ok((from.to_string(), to.to_string()))
        })
        .collect()
}

fn validate_speaker_label(label: &str) -> Result<(), String> {
    if label.is_empty() || label.chars().count() > 60 {
        return Err("El nombre del hablante no es válido.".into());
    }
    if label
        .chars()
        .any(|ch| ch.is_control() || ch == ':' || ch == '\u{E000}' || ch == '\u{E001}')
    {
        return Err(
            "El nombre del hablante no puede incluir saltos de línea ni dos puntos.".into(),
        );
    }
    Ok(())
}

fn apply_renames(text: &str, renames: &[(String, String)]) -> Result<String, String> {
    let mut current = text.to_string();
    let markers: Vec<String> = (0..renames.len())
        .map(|index| format!("\u{E000}{index}\u{E001}"))
        .collect();
    for (index, (from, to)) in renames.iter().enumerate() {
        if from == to {
            continue;
        }
        current = replace_line_label(&current, from, &markers[index]);
    }
    for (index, (_, to)) in renames.iter().enumerate() {
        current = current.replace(&format!("{}:", markers[index]), &format!("{to}:"));
    }
    Ok(current)
}

fn replace_line_label(text: &str, label: &str, marker: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (body, ending) = line_ending(line);
        if let Some(rest) = body.strip_prefix(label) {
            if rest.starts_with(':') {
                out.push_str(marker);
                out.push_str(rest);
                out.push_str(ending);
                continue;
            }
        }
        out.push_str(line);
    }
    out
}

fn line_ending(line: &str) -> (&str, &str) {
    if let Some(body) = line.strip_suffix("\r\n") {
        (body, "\r\n")
    } else if let Some(body) = line.strip_suffix('\n') {
        (body, "\n")
    } else if let Some(body) = line.strip_suffix('\r') {
        (body, "\r")
    } else {
        (line, "")
    }
}
