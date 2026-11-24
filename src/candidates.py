"""Local, explainable candidate ranking from audio, frame changes, and speech."""
import array
import math
import subprocess


def probe_duration(path: str, ffprobe: str = "ffprobe") -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError("The video has no readable duration.")
    return duration


def _sample_video(path: str, duration: float, interval: int, ffmpeg: str, on_progress=None) -> list[tuple[float, float]]:
    """Return (timestamp, normalized frame difference) at a small fixed resolution."""
    frame_size = 64 * 36
    previous = None
    scores = []
    for chunk_start in range(0, math.ceil(duration), 60):
        chunk_duration = min(60, duration - chunk_start)
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", str(chunk_start), "-t", str(chunk_duration), "-i", path, "-vf", f"fps=1/{interval},scale=64:36:force_original_aspect_ratio=decrease,pad=64:36:(ow-iw)/2:(oh-ih)/2,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
            check=True, capture_output=True,
        )
        frames = [result.stdout[i:i + frame_size] for i in range(0, len(result.stdout), frame_size) if len(result.stdout[i:i + frame_size]) == frame_size]
        for index, frame in enumerate(frames):
            change = 0.0 if previous is None else sum(abs(a - b) for a, b in zip(frame, previous)) / (len(frame) * 255)
            scores.append((min(chunk_start + index * interval, duration), change))
            previous = frame
        if on_progress:
            on_progress((chunk_start + chunk_duration) / duration)
    return scores


def _sample_audio(path: str, duration: float, ffmpeg: str, on_progress=None) -> list[tuple[float, float]]:
    """Extract low-rate mono PCM and return one-second RMS energy without audio libraries."""
    import sys
    scores = []
    for chunk_start in range(0, math.ceil(duration), 60):
        chunk_duration = min(60, duration - chunk_start)
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", str(chunk_start), "-t", str(chunk_duration), "-i", path, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"],
            check=True, capture_output=True,
        )
        samples = array.array("h")
        samples.frombytes(result.stdout[:len(result.stdout) // 2 * 2])
        if sys.byteorder != "little":
            samples.byteswap()
        block = 8000
        for index in range(math.ceil(len(samples) / block)):
            values = samples[index * block:(index + 1) * block]
            rms = math.sqrt(sum(float(x) * x for x in values) / max(1, len(values))) / 32768
            scores.append((chunk_start + index, rms))
        if on_progress:
            on_progress(min(1.0, (chunk_start + chunk_duration) / duration))
    return scores


def analyze_media(path: str, duration: float, frame_interval: int = 2, ffmpeg: str = "ffmpeg", on_progress=None) -> dict:
    """Extract only small grayscale frames and low-rate audio for local activity analysis."""
    frames = _sample_video(path, duration, frame_interval, ffmpeg,
                           (lambda value: on_progress(0.4 * value, "Sampling video frames…")) if on_progress else None)
    try:
        audio = _sample_audio(path, duration, ffmpeg,
                              (lambda value: on_progress(0.4 + 0.3 * value, "Measuring audio activity…")) if on_progress else None)
    except subprocess.CalledProcessError:
        # A video with no audio stream can still provide visual candidates.
        audio = []
    return {"frames": frames, "audio": audio}


def make_candidates(segments, analysis=None, window_seconds=30, stride_seconds=15, max_candidates=30, weights=None):
    weights = weights or {"unexpectedness": 20, "action_intensity": 20, "humor_potential": 15, "audio_excitement": 15, "visual_activity": 15, "entertainment": 15}
    duration = (analysis or {}).get("duration") or max([s["end"] for s in segments], default=window_seconds)
    duration = max(float(duration), 0.1)
    frames, audio = (analysis or {}).get("frames", []), (analysis or {}).get("audio", [])
    # Normalize robustly against the recording so a quiet game is still rankable.
    frame_top = max((v for _, v in frames), default=0.01) or 0.01
    audio_top = max((v for _, v in audio), default=0.01) or 0.01
    candidates, start = [], 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        speech = [s for s in segments if s["start"] < end and s["end"] > start]
        text = " ".join(s["text"] for s in speech).strip()
        changes = [v for t, v in frames if start <= t < end]
        levels = [v for t, v in audio if start <= t < end]
        visual = min(1.0, (sum(changes) / max(1, len(changes))) / frame_top)
        loudness = min(1.0, (sum(levels) / max(1, len(levels))) / audio_top)
        peaks = sum(1 for previous, current in zip(levels, levels[1:]) if current > max(0.025, previous * 1.8))
        audio_excitement = min(1.0, loudness * 0.65 + min(peaks, 5) / 5 * 0.35)
        words = len(text.split())
        humor = min(1.0, sum(text.lower().count(w) for w in ("haha", "lol", "no way", "what", "oh", "damn", "crazy", "insane", "wtf")) / 3)
        action = min(1.0, visual * 0.65 + audio_excitement * 0.35)
        unexpected = min(1.0, max(visual, audio_excitement) * 0.65 + (0.2 if peaks else 0) + humor * 0.15)
        entertainment = min(1.0, unexpected * .3 + action * .3 + humor * .2 + audio_excitement * .1 + visual * .1)
        values = {"unexpectedness": unexpected, "action_intensity": action, "humor_potential": humor, "audio_excitement": audio_excitement, "visual_activity": visual, "entertainment": entertainment}
        total_weight = sum(max(0.0, float(weights.get(key, 0))) for key in values)
        score = round(100 * sum(values[k] * max(0.0, float(weights.get(k, 0))) for k in values) / total_weight) if total_weight else 0
        reasons = [f"{name.replace('_', ' ').title()}: {round(value * 100)}%" for name, value in values.items()]
        if not changes:
            reasons.append("No sampled frames; visual activity is unavailable")
        if not levels:
            reasons.append("No audio samples; audio excitement is unavailable")
        if text:
            reasons.append(f"{words} transcript words provide dialogue context")
        candidates.append({"start": start, "end": end, "score": max(0, min(100, score)), "text": text or "No speech detected in this window.", "reasons": reasons, "signals": {k: round(v * 100) for k, v in values.items()}})
        start += stride_seconds
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_candidates]
