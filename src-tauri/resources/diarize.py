# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "mlx-whisper; sys_platform == 'darwin'",
#   "faster-whisper; sys_platform == 'win32'",
#   "pyannote.audio>=4.0.3",
# ]
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
# ]
# torchaudio = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Transcribe and assign speakers with pyannote.

macOS transcribes with mlx-whisper. Windows transcribes with faster-whisper
(large-v3-turbo, word timestamps). The Hugging Face token is read by the
libraries from HF_TOKEN. This script never prints that value and never
receives it as an argument.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
FASTER_WHISPER_MODEL = "large-v3-turbo"
MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
MAX_CUE_CHARS = 80
MAX_CUE_SECONDS = 6.0
# Un desfase pequeño de marcas de tiempo sigue siendo el mismo turno.
SPEAKER_GAP_SECONDS = 0.4
COMPRESSION_RATIO_LIMIT = 2.4
SAMPLE_RATE = 16000
WINDOW_PAD = 0.2
WINDOW_MERGE_GAP = 0.5
WINDOW_MAX = 30.0
# Un recorte más corto suele inventar «gracias» o «¿qué?».
MIN_WINDOW = 0.8


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    try:
        run(parse_args(argv))
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(f"Error: {message}", file=sys.stderr, flush=True)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe y diariza un audio.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--format", required=True, choices=("txt", "srt"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=WHISPER_MODEL)
    parser.add_argument("--language", default="")
    parser.add_argument("--condition-on-previous-text", default="true")
    parser.add_argument("--num-speakers", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Falta el token de Hugging Face.")

    ffmpeg = args.ffmpeg or "ffmpeg"
    if ffmpeg != "ffmpeg":
        ffmpeg_dir = str(Path(ffmpeg).resolve().parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    samples = decode_audio(ffmpeg, args.audio)
    print("Identificando hablantes…", flush=True)
    turns = identify_speakers(samples, args.num_speakers)
    print("Transcribiendo…", flush=True)
    words = transcribe(
        samples,
        args.model,
        args.language,
        as_bool(args.condition_on_previous_text),
        speech_windows(turns, samples.size / SAMPLE_RATE),
    )

    print("Combinando…", flush=True)
    groups = assign_and_group(words, turns)
    destination = output_path(args.audio, args.output_dir, args.format)
    body = render_srt(groups) if args.format == "srt" else render_txt(groups)
    destination.write_text(body, encoding="utf-8")
    for group in groups:
        text = group["text"].replace("\r", " ").replace("\n", " ")
        print(f"PARAGRAPH\t{group['speaker']}: {text}", flush=True)


def decode_audio(ffmpeg: str, audio_path: str):
    import numpy as np

    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        if len(detail) > 500:
            detail = detail[:500]
        raise RuntimeError(detail or "ffmpeg no pudo decodificar el audio.")
    samples = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise RuntimeError("El audio está vacío.")
    return samples


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "si", "sí"}


def backend_available(torch_module, name: str) -> bool:
    if name == "cuda":
        probe = getattr(torch_module, "cuda", None)
    else:
        probe = getattr(getattr(torch_module, "backends", None), name, None)
    is_available = getattr(probe, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def choose_pyannote_device(platform: str, *, mps: bool, cuda: bool) -> str:
    if platform == "darwin" and mps:
        return "mps"
    if platform == "win32" and cuda:
        return "cuda"
    return "cpu"


def choose_whisper_compute(cuda: bool) -> tuple[str, str]:
    if cuda:
        return "cuda", "float16"
    return "cpu", "int8"


def identify_speakers(samples, num_speakers: int) -> list[tuple]:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZE_MODEL)
    if pipeline is None:
        raise RuntimeError(
            "No se pudo cargar el modelo. Acepta las condiciones en " + MODEL_URL
        )
    device_name = choose_pyannote_device(
        sys.platform,
        mps=backend_available(torch, "mps"),
        cuda=backend_available(torch, "cuda"),
    )
    device = torch.device(device_name)
    try:
        pipeline.to(device)
    except Exception:
        if device.type == "cpu":
            raise
        device = torch.device("cpu")
        pipeline.to(device)
    print(f"Dispositivo: {device.type}", flush=True)

    waveform = torch.from_numpy(samples).unsqueeze(0)
    call_kwargs = {}
    if num_speakers in (2, 3, 4):
        call_kwargs["num_speakers"] = num_speakers
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        **call_kwargs,
    )
    return list(iter_speaker_turns(diarization))


def speech_windows(turns: list[tuple], duration: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    spans = []
    for start, end, _speaker in turns:
        span_start = max(0.0, float(start) - WINDOW_PAD)
        span_end = min(duration, float(end) + WINDOW_PAD)
        if span_end > span_start:
            spans.append((span_start, span_end))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + WINDOW_MERGE_GAP:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    windows = []
    for start, end in merged:
        cursor = start
        while end - cursor > WINDOW_MAX:
            windows.append((cursor, cursor + WINDOW_MAX))
            cursor += WINDOW_MAX
        if end > cursor:
            windows.append((cursor, end))
    usable = [window for window in windows if window[1] - window[0] >= MIN_WINDOW]
    if usable:
        return usable
    if windows:
        return windows
    return [(0.0, duration)]


def slice_audio(samples, start: float, end: float):
    first = max(0, int(start * SAMPLE_RATE))
    last = min(samples.size, int(end * SAMPLE_RATE))
    if last <= first:
        return samples[:0]
    return samples[first:last].copy()


def shift_words(words: list[dict], offset: float) -> list[dict]:
    if offset == 0:
        return words
    return [
        {
            "word": word["word"],
            "start": float(word["start"]) + offset,
            "end": float(word["end"]) + offset,
        }
        for word in words
    ]


def transcribe(samples, model: str, language: str, use_context: bool, windows: list[tuple]) -> list[dict]:
    # El texto anterior se ignora. Si el modelo falla con «Gracias.» o «¿Qué?»,
    # reutilizarlo sustituye el resto del audio por esa frase.
    del use_context
    if sys.platform == "darwin":
        return transcribe_mlx(samples, model or WHISPER_MODEL, language, windows)
    if sys.platform == "win32":
        return transcribe_faster(samples, model or FASTER_WHISPER_MODEL, language, windows)
    raise RuntimeError("Distinguir hablantes solo está disponible en macOS y Windows.")


def transcribe_mlx(samples, model: str, language: str, windows: list[tuple]) -> list[dict]:
    import mlx_whisper

    def decode(chunk):
        options = {
            "path_or_hf_repo": model,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "verbose": False,
        }
        if language:
            options["language"] = language
        return words_from_result(mlx_whisper.transcribe(chunk, **options))

    return transcribe_windows(samples, windows, decode)


def transcribe_faster(samples, model: str, language: str, windows: list[tuple]) -> list[dict]:
    from faster_whisper import WhisperModel

    device, compute_type = choose_whisper_compute(faster_cuda_available())
    try:
        whisper = WhisperModel(model, device=device, compute_type=compute_type)
    except Exception:
        if device == "cpu":
            raise
        device, compute_type = "cpu", "int8"
        whisper = WhisperModel(model, device=device, compute_type=compute_type)

    def decode(chunk):
        segments, _info = whisper.transcribe(
            chunk,
            language=language or None,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return words_from_faster_segments(segments)

    return transcribe_windows(samples, windows, decode)


def transcribe_windows(samples, windows: list[tuple], decode) -> list[dict]:
    words = []
    for start, end in windows:
        chunk = slice_audio(samples, start, end)
        if chunk.size == 0:
            continue
        words.extend(shift_words(decode(chunk), start))
    return words


def faster_cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def segment_attr(segment, name: str):
    if isinstance(segment, dict):
        return segment.get(name)
    return getattr(segment, name, None)


def segment_is_unreliable(segment) -> bool:
    # Solo se descarta un tramo que se repite dentro de sí mismo.
    # «Gracias.» suena seguro para el modelo; el habla real, no, y no hay que tirarla.
    ratio = segment_attr(segment, "compression_ratio")
    return ratio is not None and float(ratio) > COMPRESSION_RATIO_LIMIT


def words_from_faster_segments(segments) -> list[dict]:
    words = []
    for segment in segments:
        if segment_is_unreliable(segment):
            continue
        segment_words = getattr(segment, "words", None) or []
        if segment_words:
            for word in segment_words:
                text = str(getattr(word, "word", ""))
                if not text.strip():
                    continue
                words.append(
                    {
                        "word": text,
                        "start": float(word.start),
                        "end": float(word.end),
                    }
                )
            continue
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        words.append(
            {
                "word": f" {text}",
                "start": float(segment.start),
                "end": float(segment.end),
            }
        )
    return words


def words_from_result(result: dict) -> list[dict]:
    words = []
    for segment in result.get("segments") or []:
        if segment_is_unreliable(segment):
            continue
        segment_words = segment.get("words") or []
        if segment_words:
            for word in segment_words:
                text = str(word.get("word", ""))
                if not text.strip():
                    continue
                start = float(word.get("start", segment.get("start", 0.0)))
                end = float(word.get("end", segment.get("end", start)))
                words.append({"word": text, "start": start, "end": end})
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        words.append(
            {
                "word": f" {text}",
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
            }
        )
    return words


def iter_speaker_turns(output):
    annotation = getattr(output, "speaker_diarization", output)
    if hasattr(annotation, "itertracks"):
        for segment, _track, label in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(label)
        return
    for item in annotation:
        segment = item[0]
        label = item[-1]
        yield float(segment.start), float(segment.end), str(label)


def assign_and_group(words: list[dict], turns: list[tuple]) -> list[dict]:
    ordered = sorted(turns, key=lambda turn: (turn[0], turn[1], str(turn[2])))
    groups = []
    labels = {}
    current = None
    for word in words:
        if ordered:
            raw = speaker_for_word(word, ordered)
            if raw is None:
                continue
        else:
            raw = "SPEAKER_00"
        if raw not in labels:
            labels[raw] = f"Persona {len(labels) + 1}"
        speaker = labels[raw]
        if current is None or current["speaker"] != speaker:
            current = {"speaker": speaker, "words": []}
            groups.append(current)
        current["words"].append(word)
    for group in groups:
        group["text"] = join_words(group["words"])
    return [group for group in groups if group["text"]]


def speaker_for_word(word: dict, turns: list[tuple]):
    if not turns:
        return None
    start = float(word["start"])
    end = float(word["end"])
    if end < start:
        end = start
    best = None
    best_overlap = 0.0
    for turn_start, turn_end, speaker in turns:
        amount = overlap(start, end, float(turn_start), float(turn_end))
        if amount > best_overlap:
            best_overlap = amount
            best = speaker
    if best is not None:
        return best
    midpoint = (start + end) / 2.0

    def distance(turn):
        turn_start, turn_end, _speaker = turn
        if midpoint < turn_start:
            return turn_start - midpoint
        if midpoint > turn_end:
            return midpoint - turn_end
        return 0.0

    nearest = min(turns, key=distance)
    if distance(nearest) <= SPEAKER_GAP_SECONDS:
        return nearest[2]
    return None


def overlap(start: float, end: float, turn_start: float, turn_end: float) -> float:
    return max(0.0, min(end, turn_end) - max(start, turn_start))


def join_words(words: list[dict]) -> str:
    return "".join(str(word["word"]) for word in words).strip()


def render_txt(groups: list[dict]) -> str:
    blocks = [f"{group['speaker']}: {group['text']}" for group in groups]
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def render_srt(groups: list[dict]) -> str:
    cues = []
    for group in groups:
        cues.extend(split_cues(group))
    blocks = []
    for index, (start, end, speaker, text) in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{speaker}: {text}\n"
        )
    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def split_cues(group: dict) -> list[tuple]:
    cues = []
    bucket = []
    for word in group["words"]:
        if bucket and should_split(bucket, word):
            cue = cue_from(group["speaker"], bucket)
            if cue[3]:
                cues.append(cue)
            bucket = []
        bucket.append(word)
    if bucket:
        cue = cue_from(group["speaker"], bucket)
        if cue[3]:
            cues.append(cue)
    return cues


def should_split(bucket: list[dict], word: dict) -> bool:
    prospective = join_words(bucket + [word])
    start = float(bucket[0]["start"])
    end = float(word["end"])
    if len(prospective) > MAX_CUE_CHARS:
        return True
    spoken = join_words(bucket).rstrip()
    ended = bool(spoken) and spoken[-1] in ".!?…"
    duration = end - start
    if duration > MAX_CUE_SECONDS and ended:
        return True
    return duration > MAX_CUE_SECONDS + 4


def cue_from(speaker: str, bucket: list[dict]) -> tuple:
    start = float(bucket[0]["start"])
    end = float(bucket[-1]["end"])
    if end <= start:
        end = start + 0.2
    return (start, end, speaker, join_words(bucket))


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def output_path(audio: str, directory: str, extension: str) -> Path:
    name = Path(audio).name
    dot = name.rfind(".")
    stem = name[:dot] if dot > 0 else name
    return Path(directory) / f"{stem}.{extension}"


def self_test() -> int:
    failures = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    words = [
        {"word": " Hola", "start": 0.1, "end": 0.5},
        {"word": " mundo", "start": 0.6, "end": 1.2},
        {"word": " adiós", "start": 2.2, "end": 2.8},
    ]
    groups = assign_and_group(
        words,
        [(0.0, 2.0, "SPEAKER_01"), (2.0, 4.0, "SPEAKER_00")],
    )
    check("two speakers", [group["speaker"] for group in groups] == ["Persona 1", "Persona 2"])
    check("first text", groups[0]["text"] == "Hola mundo")
    check("second text", groups[1]["text"] == "adiós")

    tied = speaker_for_word(
        {"word": " a", "start": 1.0, "end": 3.0},
        [(0.0, 2.0, "early"), (2.0, 4.0, "late")],
    )
    check("tie keeps earlier speaker", tied == "early")

    nearest = speaker_for_word(
        {"word": " z", "start": 5.0, "end": 5.2},
        [(0.0, 1.0, "A"), (10.0, 11.0, "B")],
    )
    check("far gap is dropped", nearest is None)
    close = speaker_for_word(
        {"word": " z", "start": 1.1, "end": 1.3},
        [(0.0, 1.0, "A"), (10.0, 11.0, "B")],
    )
    check("near gap keeps speaker", close == "A")
    skipped = assign_and_group(
        [
            {"word": " Hola", "start": 0.1, "end": 0.4},
            {"word": " Gracias.", "start": 5.0, "end": 5.2},
        ],
        [(0.0, 1.0, "A")],
    )
    check(
        "silence is not attached",
        len(skipped) == 1 and skipped[0]["text"] == "Hola",
    )

    mostly_second = speaker_for_word(
        {"word": " x", "start": 1.8, "end": 2.6},
        [(0.0, 2.0, "A"), (2.0, 4.0, "B")],
    )
    check("larger overlap wins", mostly_second == "B")

    long_words = [
        {"word": f" palabra{index:02d}", "start": index * 0.5, "end": index * 0.5 + 0.4}
        for index in range(40)
    ]
    long_groups = assign_and_group(long_words, [(0.0, 30.0, "S")])
    txt = render_txt(long_groups)
    srt = render_srt(long_groups)
    check("txt is one paragraph", txt.count("Persona 1:") == 1 and txt.endswith("\n"))
    check("srt splits cues", srt.count("Persona 1:") >= 2)
    check("srt timestamp", "-->" in srt and ",000" in srt)
    check("srt numbers cues", srt.startswith("1\n"))

    mixed = render_txt(groups)
    check(
        "txt labels",
        mixed == "Persona 1: Hola mundo\n\nPersona 2: adiós\n",
    )

    class Segment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class Tracks:
        def itertracks(self, yield_label: bool = False):
            assert yield_label
            yield Segment(0.2, 1.5), 0, "SPEAKER_00"

    class Pairs:
        def __iter__(self):
            yield Segment(1.0, 2.0), "SPEAKER_07"

    class Boxed:
        def __init__(self, annotation) -> None:
            self.speaker_diarization = annotation

    check(
        "itertracks",
        list(iter_speaker_turns(Boxed(Tracks()))) == [(0.2, 1.5, "SPEAKER_00")],
    )
    check(
        "pair iterator",
        list(iter_speaker_turns(Boxed(Pairs()))) == [(1.0, 2.0, "SPEAKER_07")],
    )

    recovered = words_from_result(
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "sin palabras",
                    "words": [{"word": "   ", "start": 0.0, "end": 0.2}],
                },
                {"start": 1.0, "end": 2.0, "text": " fallback "},
            ]
        }
    )
    check("fallback word", recovered == [{"word": " fallback", "start": 1.0, "end": 2.0}])
    check("stem", output_path("/tmp/entrevista.final.wav", "/tmp", "txt") == Path("/tmp/entrevista.final.txt"))
    check("mac uses mps", choose_pyannote_device("darwin", mps=True, cuda=False) == "mps")
    check("mac without mps uses cpu", choose_pyannote_device("darwin", mps=False, cuda=True) == "cpu")
    check("windows uses cuda", choose_pyannote_device("win32", mps=False, cuda=True) == "cuda")
    check("windows without gpu uses cpu", choose_pyannote_device("win32", mps=False, cuda=False) == "cpu")
    class DummyTorch:
        pass

    check("missing mps", backend_available(DummyTorch(), "mps") is False)
    check("missing cuda", backend_available(DummyTorch(), "cuda") is False)
    check("whisper cuda", choose_whisper_compute(True) == ("cuda", "float16"))
    check("whisper cpu", choose_whisper_compute(False) == ("cpu", "int8"))

    class Word:
        def __init__(self, word: str, start: float, end: float) -> None:
            self.word = word
            self.start = start
            self.end = end

    class FasterSegment:
        def __init__(self, text: str, start: float, end: float, words=None) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.words = words

    faster_words = words_from_faster_segments(
        [
            FasterSegment("hola", 0.0, 1.0, [Word(" hola", 0.1, 0.4), Word("   ", 0.5, 0.6)]),
            FasterSegment(" sin marcas ", 1.0, 2.0, None),
        ]
    )
    check(
        "faster words",
        faster_words
        == [
            {"word": " hola", "start": 0.1, "end": 0.4},
            {"word": " sin marcas", "start": 1.0, "end": 2.0},
        ],
    )

    class ScoredSegment:
        def __init__(self, text, start, end, words, compression_ratio, no_speech_prob, avg_logprob):
            self.text = text
            self.start = start
            self.end = end
            self.words = words
            self.compression_ratio = compression_ratio
            self.no_speech_prob = no_speech_prob
            self.avg_logprob = avg_logprob

    filtered = words_from_faster_segments(
        [
            ScoredSegment("hola", 0.0, 1.0, [Word(" hola", 0.1, 0.4)], 1.1, 0.1, -0.2),
            ScoredSegment(
                "Gracias. Gracias. Gracias. Gracias.",
                1.0,
                4.0,
                [Word(" Gracias.", 1.0, 1.2)],
                3.6,
                0.1,
                -0.2,
            ),
            ScoredSegment("ruido", 4.0, 5.0, [Word(" ruido", 4.1, 4.4)], 1.2, 0.9, -1.4),
        ]
    )
    check("unreliable segments dropped", filtered == [
        {"word": " hola", "start": 0.1, "end": 0.4},
        {"word": " ruido", "start": 4.1, "end": 4.4},
    ])

    check("no turns uses the file", speech_windows([], 10) == [(0.0, 10.0)])
    check(
        "nearby turns merge",
        speech_windows([(1.0, 2.0, "A"), (2.2, 3.0, "B")], 10) == [(0.8, 3.2)],
    )
    check(
        "distant turns stay apart",
        speech_windows([(1.0, 3.0, "A"), (8.0, 9.0, "B")], 12) == [(0.8, 3.2), (7.8, 9.2)],
    )
    check(
        "long speech splits",
        speech_windows([(0.0, 75.0, "A")], 80) == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.2)],
    )
    check(
        "tiny blip is skipped when other speech exists",
        speech_windows([(1.0, 6.0, "A"), (20.0, 20.2, "B")], 30) == [(0.8, 6.2)],
    )

    if failures:
        print("Error: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
