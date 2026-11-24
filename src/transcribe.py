from functools import lru_cache


@lru_cache(maxsize=1)
def _get_model(model_size: str, device: str, compute_type: str):
    # Keep at most one Whisper model resident across Streamlit reruns.
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_video(path: str, model_size: str = "base", device: str = "cpu", compute_type: str = "int8", on_progress=None):
    model = _get_model(model_size, device, compute_type)
    segments, info = model.transcribe(path, word_timestamps=False, vad_filter=True)
    result = []
    for s in segments:
        result.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
        if on_progress:
            on_progress(float(s.end))
    return result, info.language
