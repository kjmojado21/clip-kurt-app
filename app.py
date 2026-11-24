import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from src.candidates import analyze_media, make_candidates, probe_duration
from src.render import render_vertical
from src.transcribe import transcribe_video

ROOT = Path(__file__).parent
with (ROOT / "config" / "settings.json").open(encoding="utf-8") as handle:
    DEFAULTS = json.load(handle)

st.set_page_config(page_title="GTA VI Clip Factory", page_icon="🎮", layout="wide")
st.title("🎮 GTA VI Clip Factory")
st.caption("Local-first highlight discovery and vertical exports. Video is processed on this computer.")

with st.sidebar:
    st.header("Detection settings")
    window = st.slider("Candidate window (seconds)", 10, 60, int(DEFAULTS["clip_duration_default_seconds"]), 5)
    stride = st.slider("Scan stride (seconds)", 5, 30, min(15, window), 5)
    frame_interval = st.slider("Frame sample interval (seconds)", 1, 10, int(DEFAULTS["frame_sample_interval_seconds"]))
    max_candidates = st.slider("Maximum candidates", 1, 50, int(DEFAULTS["max_candidates"]))
    transcribe = st.checkbox("Transcribe dialogue locally (Whisper)", True)
    st.caption(f"Local Whisper model: `{DEFAULTS['transcription_model']}` · {DEFAULTS['transcription_device']}")
    st.subheader("Score weights")
    weights = {key: st.slider(label, 0, 40, int(DEFAULTS["score_weights"][key]), 1, key=f"weight_{key}")
               for key, label in [("unexpectedness", "Unexpectedness"), ("action_intensity", "Action intensity"),
                                  ("humor_potential", "Humor potential"), ("audio_excitement", "Audio excitement"),
                                  ("visual_activity", "Visual activity"), ("entertainment", "Overall entertainment") ]}

uploaded = st.file_uploader("Import a gameplay recording", type=["mp4", "mov", "mkv", "webm"])
analyze_clicked = st.button("Analyze gameplay for highlights", type="primary", disabled=uploaded is None)
if uploaded is None:
    st.caption("Upload a gameplay video above to enable analysis.")
if uploaded:
    digest = hashlib.sha256(uploaded.getbuffer()).hexdigest()
    if st.session_state.get("upload_digest") != digest:
        old_path = st.session_state.get("source")
        if old_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except OSError:
                pass
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            st.session_state["source"] = tmp.name
        st.session_state["upload_digest"] = digest
        st.session_state.pop("candidates", None)
    source = st.session_state["source"]
    st.video(uploaded)
    st.info("The source is copied to a temporary local file for this session. The original file is never modified or uploaded by this app.")
    try:
        video_duration = probe_duration(source)
        st.caption(f"Video duration: {video_duration:.1f} seconds")
    except Exception as exc:
        video_duration = 0
        st.error(f"Could not read video metadata. Check that FFmpeg and FFprobe are available: {exc}")

    if analyze_clicked:
        if not video_duration:
            st.error("Analysis cannot start because FFprobe could not read this video's duration. Check FFmpeg/FFprobe installation and the file format.")
        else:
            progress = st.progress(0, text="Preparing local media analysis…")
            try:
                def media_progress(fraction, label):
                    progress.progress(min(45, max(1, int(45 * fraction))), text=label)
                signals = analyze_media(source, video_duration, frame_interval, on_progress=media_progress)
                segments, language = [], "not transcribed"
                if transcribe:
                    def speech_progress(seconds):
                        progress.progress(min(88, 48 + int(40 * seconds / max(video_duration, 1))), text=f"Transcribing locally… {min(100, int(100 * seconds / video_duration))}%")
                    segments, language = transcribe_video(source, model_size=DEFAULTS["transcription_model"], device=DEFAULTS["transcription_device"], compute_type=DEFAULTS["transcription_compute_type"], on_progress=speech_progress)
                progress.progress(90, text="Scoring candidate moments…")
                signals["duration"] = video_duration
                st.session_state["segments"] = segments
                st.session_state["language"] = language
                st.session_state["signals"] = signals
                st.session_state["candidates"] = make_candidates(segments, signals, window, stride, max_candidates, weights)
                st.session_state["analyzed_settings"] = {"window": window, "stride": stride, "frame_interval": frame_interval, "transcribe": transcribe}
                progress.progress(100, text="Analysis complete")
            except Exception as exc:
                progress.empty()
                st.error(f"Analysis failed: {exc}. Confirm FFmpeg/FFprobe work and that the selected model is installed locally.")

if st.session_state.get("candidates"):
    st.write(f"Speech language: **{st.session_state.get('language', 'unknown')}** · candidates are ranked from local audio, sampled frame activity, and optional transcript cues.")
    candidates = st.session_state["candidates"]
    st.subheader("Review candidate moments")
    for index, candidate in enumerate(candidates):
        with st.expander(f"#{index + 1} · {candidate['start']:.1f}–{candidate['end']:.1f}s · {candidate['score']}/100", expanded=index == 0):
            st.write(candidate["text"])
            st.write("**Score reasons**")
            st.write(" · ".join(candidate["reasons"]))
            left, right = st.columns(2)
            with left:
                start = st.number_input("Start (seconds)", min_value=0.0, max_value=max(0.0, video_duration - 0.1), value=min(float(candidate["start"]), max(0.0, video_duration - 0.1)), step=0.5, key=f"start_{index}")
            with right:
                end_default = min(float(candidate["end"]), video_duration)
                end = st.number_input("End (seconds)", min_value=0.1, max_value=max(0.1, video_duration), value=max(0.1, end_default), step=0.5, key=f"end_{index}")
            captions = st.checkbox("Burn in captions when dialogue is present", value=True, key=f"captions_{index}")
            caption_size = st.slider("Caption size", 10, 22, 14, key=f"caption_size_{index}") if captions else 14
            caption_position = st.selectbox("Caption position", ["Bottom", "Top"], key=f"caption_position_{index}") if captions else "Bottom"
            crop_x = st.slider("Horizontal framing (left to right)", 0, 100, 50, key=f"crop_x_{index}", help="Shift the vertical crop toward the player or action you want to keep visible.")
            st.video(source, start_time=max(0, int(start)))
            if st.button("Render vertical 9:16 clip", key=f"render_{index}"):
                if end <= start:
                    st.error("End time must be after start time.")
                else:
                    output = ROOT / "exports" / f"clip_{index + 1}.mp4"
                    try:
                        with st.spinner("Rendering with FFmpeg…"):
                            render_vertical(source, start, end, str(output), st.session_state.get("segments", []), captions,
                                            DEFAULTS["output_width"], DEFAULTS["output_height"],
                                            caption_size=caption_size, caption_position=caption_position, crop_x=crop_x)
                        st.success(f"Rendered: {output}")
                        st.video(str(output))
                        with output.open("rb") as clip:
                            st.download_button("Download clip", clip, file_name=output.name, mime="video/mp4", key=f"download_{index}")
                    except Exception as exc:
                        st.error(f"Export failed: {exc}. Confirm FFmpeg includes libx264 and subtitles support.")
else:
    st.markdown("**Workflow:** import → sample audio and frames → optionally transcribe → review scored windows → adjust timestamps → render a vertical clip.")
