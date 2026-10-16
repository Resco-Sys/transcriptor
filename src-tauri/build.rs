fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "start_transcription",
            "flatten_transcript",
            "apply_speaker_names",
            "save_hf_token",
            "hf_token_status",
            "test_hf_token",
        ]),
    ))
    .expect("failed to run tauri-build");
}
