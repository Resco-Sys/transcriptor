use std::time::Duration;

const SERVICE: &str = "com.ivan.transcriptor";
const TOKEN_USER: &str = "huggingface";
const VALID_USER: &str = "huggingface-valid";
const VALID_MARK: &str = "ok";

fn store_name() -> &'static str {
    if cfg!(windows) {
        "Administrador de credenciales"
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
    entry(TOKEN_USER)?.set_password(&token).map_err(|error| {
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

fn entry(user: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE, user)
        .map_err(|error| format!("No se pudo abrir el {}: {error}", store_name()))
}

fn read_secret(user: &str) -> Result<Option<String>, String> {
    match entry(user)?.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(format!("No se pudo leer el {}: {error}", store_name())),
    }
}

fn mark_valid() -> Result<(), String> {
    entry(VALID_USER)?
        .set_password(VALID_MARK)
        .map_err(|error| {
            format!(
                "No se pudo guardar el estado en el {}: {error}",
                store_name()
            )
        })
}

fn clear_valid() -> Result<(), String> {
    match entry(VALID_USER)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(format!(
            "No se pudo actualizar el {}: {error}",
            store_name()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Http, interpret, normalize_token};

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

    #[test]
    fn maps_hugging_face_responses() {
        assert_eq!(interpret(Http::Status(200), Http::Status(200)), "ok");
        assert_eq!(interpret(Http::Status(401), Http::Failed), "invalid");
        assert_eq!(interpret(Http::Status(200), Http::Status(403)), "gated");
        assert_eq!(interpret(Http::Failed, Http::Status(200)), "network");
        assert_eq!(interpret(Http::Status(200), Http::Status(404)), "network");
    }
}
