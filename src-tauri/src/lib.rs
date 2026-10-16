mod secrets;
mod transcribe;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            transcribe::start_transcription,
            transcribe::flatten_transcript,
            transcribe::apply_speaker_names,
            secrets::save_hf_token,
            secrets::hf_token_status,
            secrets::test_hf_token
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
