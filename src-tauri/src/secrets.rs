use std::time::Duration;

const SERVICE: &str = "com.ivan.transcriptor";
const TOKEN_USER: &str = "huggingface";
const VALID_USER: &str = "huggingface-valid";
const VALID_MARK: &str = "ok";

fn store_name() -> &'static str {
    if cfg!(windows) {
        "Administrador de credenciales"
    } else if cfg!(target_os = "macos") {
        "almacén de la app"
    } else {
        "Llavero"
    }
}
const DIARIZE_MODEL: &str = "pyannote/speaker-diarization-community-1";
const WHOAMI_URL: &str = "https://huggingface.co/api/whoami-v2";

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenStatus {
    saved: bool,
    valid: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenTest {
    ok: bool,
    code: String,
}

#[tauri::command]
pub fn save_hf_token(token: String) -> Result<(), String> {
    let token = normalize_token(&token)?;
    clear_valid()?;
    write_secret(TOKEN_USER, &token).map_err(|error| {
        format!(
            "No se pudo guardar el token en el {}: {error}",
            store_name()
        )
    })
}

#[tauri::command]
pub fn hf_token_status() -> Result<TokenStatus, String> {
    let saved = read_secret(TOKEN_USER)?.is_some_and(|token| !token.is_empty());
    let valid = saved && read_secret(VALID_USER)?.as_deref() == Some(VALID_MARK);
    Ok(TokenStatus { saved, valid })
}

#[tauri::command]
pub fn test_hf_token() -> Result<TokenTest, String> {
    let Some(token) = read_secret(TOKEN_USER)? else {
        clear_valid()?;
        return Ok(TokenTest {
            ok: false,
            code: "missing".into(),
        });
    };
    if token.is_empty() {
        clear_valid()?;
        return Ok(TokenTest {
            ok: false,
            code: "missing".into(),
        });
    }

    let code = probe(&token);
    if code == "ok" {
        let current = read_secret(TOKEN_USER)?;
        if current.as_deref() == Some(token.as_str()) {
            mark_valid()?;
        }
    } else if code == "invalid" || code == "gated" || code == "missing" {
        clear_valid()?;
    }
    Ok(TokenTest {
        ok: code == "ok",
        code: code.into(),
    })
}

pub fn require_valid_token() -> Result<String, String> {
    let Some(token) = read_secret(TOKEN_USER)? else {
        return Err("No hay un token de Hugging Face guardado.".into());
    };
    if token.is_empty() {
        return Err("No hay un token de Hugging Face guardado.".into());
    }
    if read_secret(VALID_USER)?.as_deref() != Some(VALID_MARK) {
        return Err("Prueba el token en Configuración antes de distinguir hablantes.".into());
    }
    Ok(token)
}

fn probe(token: &str) -> &'static str {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(20))
        .build();
    let whoami = http_status(&agent, WHOAMI_URL, token);
    if !matches!(whoami, Http::Status(200)) {
        return interpret(whoami, Http::Failed);
    }
    let auth = http_status(&agent, &auth_check_url(), token);
    let model = if matches!(auth, Http::Status(404)) {
        http_status(&agent, &resolve_url(), token)
    } else {
        auth
    };
    interpret(Http::Status(200), model)
}

fn auth_check_url() -> String {
    format!("https://huggingface.co/api/models/{DIARIZE_MODEL}/auth-check")
}

fn resolve_url() -> String {
    format!("https://huggingface.co/{DIARIZE_MODEL}/resolve/main/config.yaml")
}

#[derive(Clone, Copy)]
enum Http {
    Status(u16),
    Failed,
}

fn interpret(whoami: Http, model: Http) -> &'static str {
    match whoami {
        Http::Failed => "network",
        Http::Status(200) => match model {
            Http::Status(200) => "ok",
            Http::Status(401) => "invalid",
            Http::Status(403) => "gated",
            Http::Failed => "network",
            _ => "network",
        },
        Http::Status(401) | Http::Status(403) => "invalid",
        _ => "network",
    }
}

fn http_status(agent: &ureq::Agent, url: &str, token: &str) -> Http {
    let authorization = format!("Bearer {token}");
    match agent
        .get(url)
        .set("Authorization", &authorization)
        .set("User-Agent", "transcriptor")
        .call()
    {
        Ok(response) => Http::Status(response.status()),
        Err(ureq::Error::Status(code, _)) => Http::Status(code),
        Err(_) => Http::Failed,
    }
}

fn normalize_token(raw: &str) -> Result<String, String> {
    let mut token = raw.trim().to_string();
    if token.len() >= 2 {
        let quote = token.as_bytes()[0];
        let last = token.as_bytes()[token.len() - 1];
        if (quote == b'"' || quote == b'\'') && quote == last {
            token = token[1..token.len() - 1].trim().to_string();
        }
    }
    if let Some(rest) = strip_bearer(&token) {
        token = rest.trim().to_string();
    }
    if token.len() < 10
        || token.len() > 500
        || token
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        return Err("El token no es válido.".into());
    }
    Ok(token)
}

fn strip_bearer(token: &str) -> Option<&str> {
    let rest = token.get(7..)?;
    token
        .get(..7)?
        .eq_ignore_ascii_case("bearer ")
        .then_some(rest)
}

#[cfg(not(target_os = "macos"))]
fn entry(user: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE, user)
        .map_err(|error| format!("No se pudo abrir el {}: {error}", store_name()))
}

fn read_secret(user: &str) -> Result<Option<String>, String> {
    // El Llavero de inicio de sesión pide la contraseña en cada arranque
    // cuando la firma de la app cambia. En macOS el token va a un archivo.
    #[cfg(target_os = "macos")]
    {
        return read_file_secret(&secrets_dir()?, user)
            .map_err(|error| format!("No se pudo leer el {}: {error}", store_name()));
    }
    #[cfg(not(target_os = "macos"))]
    {
        match entry(user)?.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(format!("No se pudo leer el {}: {error}", store_name())),
        }
    }
}

fn write_secret(user: &str, value: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        return write_file_secret(&secrets_dir()?, user, value);
    }
    #[cfg(not(target_os = "macos"))]
    {
        entry(user)?
            .set_password(value)
            .map_err(|error| error.to_string())
    }
}

fn delete_secret(user: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        return delete_file_secret(&secrets_dir()?, user);
    }
    #[cfg(not(target_os = "macos"))]
    {
        match entry(user)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }
}

fn mark_valid() -> Result<(), String> {
    write_secret(VALID_USER, VALID_MARK).map_err(|error| {
        format!(
            "No se pudo guardar el estado en el {}: {error}",
            store_name()
        )
    })
}

fn clear_valid() -> Result<(), String> {
    delete_secret(VALID_USER)
        .map_err(|error| format!("No se pudo actualizar el {}: {error}", store_name()))
}

/// `~/Library/Application Support/com.ivan.transcriptor/secrets`
#[cfg(target_os = "macos")]
fn secrets_dir() -> Result<std::path::PathBuf, String> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .ok_or("No se encontró la carpeta del usuario.")?;
    Ok(std::path::PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join(SERVICE)
        .join("secrets"))
}

#[cfg(any(test, target_os = "macos"))]
fn validate_secret_name(user: &str) -> Result<(), String> {
    if user.is_empty()
        || user.contains('/')
        || user.contains('\\')
        || user.contains("..")
        || user.contains('\0')
    {
        return Err("Nombre de secreto no válido.".into());
    }
    Ok(())
}

#[cfg(any(test, target_os = "macos"))]
fn read_file_secret(dir: &std::path::Path, user: &str) -> Result<Option<String>, String> {
    validate_secret_name(user)?;
    match std::fs::read(dir.join(user)) {
        Ok(bytes) => {
            let text = String::from_utf8(bytes)
                .map_err(|_| "El dato guardado no se puede leer.".to_string())?;
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(any(test, target_os = "macos"))]
fn write_file_secret(dir: &std::path::Path, user: &str, value: &str) -> Result<(), String> {
    validate_secret_name(user)?;
    std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    restrict_to_owner(dir)?;
    let path = dir.join(user);
    let tmp = dir.join(format!(".{user}.tmp"));
    if let Err(error) = write_private(&tmp, value.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    #[cfg(windows)]
    if path.exists() {
        if let Err(error) = std::fs::remove_file(&path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(error.to_string());
        }
    }
    if let Err(error) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error.to_string());
    }
    Ok(())
}

#[cfg(any(test, target_os = "macos"))]
fn delete_file_secret(dir: &std::path::Path, user: &str) -> Result<(), String> {
    validate_secret_name(user)?;
    match std::fs::remove_file(dir.join(user)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(any(test, target_os = "macos"))]
fn restrict_to_owner(dir: &std::path::Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(dir)
            .map_err(|error| error.to_string())?
            .permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(dir, permissions).map_err(|error| error.to_string())?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

#[cfg(any(test, target_os = "macos"))]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    file.write_all(bytes).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = file
            .metadata()
            .map_err(|error| error.to_string())?
            .permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(path, permissions).map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Http, delete_file_secret, interpret, normalize_token, read_file_secret, write_file_secret,
    };

    #[test]
    fn accepts_a_plain_token_and_strips_wrapping() {
        assert_eq!(
            normalize_token("  \"Bearer hf_abcdefghij\"  ").unwrap(),
            "hf_abcdefghij"
        );
    }

    #[test]
    fn rejects_short_or_spaced_tokens() {
        assert!(normalize_token("short").is_err());
        assert!(normalize_token("hf_abc defghij").is_err());
    }

    #[cfg(windows)]
    #[test]
    fn windows_credential_manager_keeps_the_token() {
        let entry =
            keyring::Entry::new("com.ivan.transcriptor.test", "probe").expect("credential entry");
        let _ = entry.delete_credential();
        entry.set_password("hf_roundtrip").expect("set password");
        let stored = entry.get_password().expect("get password");
        let _ = entry.delete_credential();
        assert_eq!(stored, "hf_roundtrip");
    }

    #[test]
    fn file_store_keeps_the_token_private() {
        let dir =
            std::env::temp_dir().join(format!("transcriptor-secret-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_file_secret(&dir, "huggingface", "hf_roundtrip").expect("guardar");
        write_file_secret(&dir, "huggingface", "hf_replaced").expect("reemplazar");
        assert_eq!(
            read_file_secret(&dir, "huggingface")
                .expect("leer")
                .as_deref(),
            Some("hf_replaced")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let file_mode = std::fs::metadata(dir.join("huggingface"))
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            let dir_mode = std::fs::metadata(&dir)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(file_mode, 0o600);
            assert_eq!(dir_mode, 0o700);
        }
        delete_file_secret(&dir, "huggingface-valid").expect("borrar ausente");
        delete_file_secret(&dir, "huggingface").expect("borrar");
        assert_eq!(
            read_file_secret(&dir, "huggingface").expect("leer vacío"),
            None
        );
        assert!(write_file_secret(&dir, "../huggingface", "hf_roundtrip").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn maps_hugging_face_responses() {
        assert_eq!(interpret(Http::Status(200), Http::Status(200)), "ok");
        assert_eq!(interpret(Http::Status(401), Http::Failed), "invalid");
        assert_eq!(interpret(Http::Status(200), Http::Status(403)), "gated");
        assert_eq!(interpret(Http::Failed, Http::Status(200)), "network");
        assert_eq!(interpret(Http::Status(200), Http::Status(404)), "network");
    }
}
