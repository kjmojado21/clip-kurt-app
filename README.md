# GTA VI Clip Factory

A local-first Streamlit tool for finding possible highlights in gameplay recordings you own or may edit. It does not send footage to an external service and never writes into the source video.

## What it does

- Imports MP4, MOV, MKV, and WebM recordings.
- Uses FFprobe for duration, then FFmpeg to sample low-resolution grayscale frames and low-rate mono audio in bounded chunks.
- Optionally transcribes dialogue with the configured local `faster-whisper` model (`base`, CPU, int8 by default).
- Ranks overlapping candidate windows from visual frame changes, audio energy/spikes, and dialogue excitement cues. Each score includes six 0–100 signal values and explanations.
- Lets you adjust candidate timestamps and preview the source at those points.
- Renders 9:16 H.264/AAC MP4 with FFmpeg. Captions are split into short timed phrases, can be moved to the top or bottom, and have an adjustable size. The horizontal crop can be shifted to keep the player/action in frame.
- Saves exports under `exports/`; the uploaded source is held in a temporary local session file.

The detection signals are activity proxies. Without a configured local video-understanding model, the app cannot reliably identify semantic events such as a police chase, crash, explosion, NPC gag, or gunfight. A loud or visually busy segment is only a candidate and requires review.

## Windows setup and launch

Install Python 3.11+ and FFmpeg/FFprobe, and make sure `ffmpeg -version` and `ffprobe -version` work in PowerShell. From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the virtual environment already exists, activate it and run the final three commands. Open the local URL Streamlit prints (usually `http://localhost:8501`). The first Whisper run may download the model from its model repository; subsequent inference is local. Disable transcription to analyze without downloading/loading Whisper.

## Sample-video test procedure

1. Use a short gameplay recording with a mix of quiet travel and at least one busy/loud event. The app does not include or download sample footage.
2. Launch the app and upload the recording. Confirm duration is shown and try **Analyze gameplay for highlights**. For an audio/visual-only smoke check, disable local transcription first.
3. Review the candidate windows, scores, reasons, and timestamps. Change a start/end time and preview from the revised start.
4. Render a short window with captions enabled, then download/open the MP4. Confirm it is portrait 9:16, the selected range is present, captions appear only when transcript segments overlap, and the original recording remains unchanged.
5. Repeat with transcription disabled and a silent or low-audio sample. No speech should be required to produce visual-activity candidates.

## Dependencies and limits

Python dependencies are listed in `requirements.txt`: Streamlit and faster-whisper. FFmpeg/FFprobe are external system executables. No Python image-analysis package is required. Audio and frame analysis are lightweight heuristics, not game-event recognition; score weights and sampling/window settings are editable in the sidebar. CPU transcription can be slow. `libx264` and FFmpeg's `subtitles` filter are needed for encoding and burned captions. Vertical framing is center-cropped and may cut off action near the sides. Review each result before publishing.
