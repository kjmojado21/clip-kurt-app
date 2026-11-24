import subprocess
import tempfile
from pathlib import Path


def _srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def _caption_chunks(text: str, limit: int = 32) -> list[str]:
    words = text.split()
    chunks, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > limit:
            chunks.append(line)
            line = word
        else:
            line = candidate
    if line:
        chunks.append(line)
    return chunks


def render_vertical(source: str, start: float, end: float, output: str, segments=None, captions=True, width=1080, height=1920, ffmpeg="ffmpeg", caption_size=14, caption_position="Bottom", crop_x=50):
    if start < 0 or end <= start:
        raise ValueError("Clip end must be after its non-negative start time.")
    duration = end - start
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    # Scale to fill before center-cropping; handles both landscape and portrait input.
    crop_fraction = min(100, max(0, int(crop_x))) / 100
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}:(iw-ow)*{crop_fraction:.2f}:0"
    with tempfile.TemporaryDirectory(prefix="gta6_captions_") as tmp:
        if captions and segments:
            selected = [s for s in segments if s["start"] < end and s["end"] > start and s["text"].strip()]
            if selected:
                srt = Path(tmp) / "captions.srt"
                blocks = []
                index = 1
                for segment in selected:
                    left = max(0, segment["start"] - start)
                    right = min(duration, segment["end"] - start)
                    if right > left:
                        chunks = _caption_chunks(segment["text"])
                        total_chars = sum(len(chunk) for chunk in chunks) or 1
                        elapsed = 0.0
                        for chunk in chunks:
                            chunk_duration = (right - left) * len(chunk) / total_chars
                            chunk_start, chunk_end = left + elapsed, left + elapsed + chunk_duration
                            blocks.append(f"{index}\n{_srt_time(chunk_start)} --> {_srt_time(chunk_end)}\n{chunk}\n")
                            elapsed += chunk_duration
                            index += 1
                if blocks:
                    srt.write_text("\n".join(blocks), encoding="utf-8")
                    subtitle_path = srt.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
                    alignment = 8 if caption_position == "Top" else 2
                    margin = 160 if alignment == 2 else 120
                    style = f"FontName=Arial,FontSize={int(caption_size)},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,BackColour=&H90000000,Outline=0,Shadow=0,Alignment={alignment},MarginV={margin},MarginL=100,MarginR=100"
                    vf += f",subtitles=filename='{subtitle_path}':force_style='{style}'"
        cmd = [ffmpeg, "-hide_banner", "-y", "-ss", str(start), "-i", source, "-t", str(duration), "-vf", vf,
               "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", output]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output
