# 🎙️ Singing Similarity Evaluator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-green)](https://librosa.org/)

A interactive tool designed to evaluate singing accuracy by comparing a user's recording against a reference track. It utilizes **Dynamic Time Warping (DTW)** and **Chroma Feature Analysis** to provide detailed feedback on pitch and tempo, complete with visual charts and a generated animation video.

---

## ✨ Key Features

* **🎤 Dual Input Source:** Upload audio files or record directly via microphone.
* **🧩 Robust Alignment:** Uses **DTW (Dynamic Time Warping)** to align the user's singing with the reference track, handling timing variations gracefully.
* **📊 Windowed Analysis:** detailed evaluation using a sliding window approach (3s window, 1s overlap) to identify specific segments where pitch or tempo drifts.
* **📈 Visual Feedback:**
    * **Interactive Charts:** Plot pitch, tempo, and overall scores over time.
    * **Dynamic Animation:** Generates an `.mp4` video showing real-time pitch deviation (Sharp/Flat/Accurate) synchronized with the audio.
* **🎧 Mixed Playback:** Automatically mixes the reference and user audio for easy aural comparison.
* **📝 Detailed Report:** Provides textual feedback identifying the "Best Segment" and "Segments Needing Improvement".

---

## 🛠️ Prerequisites

To run this project, you need **Python 3.8+** and **FFmpeg** installed on your system.

### 1. System Dependency: FFmpeg
The tool uses `ffmpeg` to merge the generated animation frames with audio.

* **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your System PATH.
* **macOS:** `brew install ffmpeg`
* **Linux (Ubuntu/Debian):** `sudo apt install ffmpeg`

### 2. Python Dependencies
Install the required libraries:

```bash
pip install -r requirements.txt
```
## 🚀 Usage

1. **Clone the repository** (or save the script):
   ```bash
   git clone https://github.com/yourusername/singing-evaluator.git
   cd singing-evaluator
   ```

2. **Run the application**:
    ```bash
    python singing_evaluator.py
   ```

3. **Access the Interface**:
    The terminal will output a local URL (usually http://127.0.0.1:7860). Open this link in your browser.


4. **Start Evaluation**:

    - Input: Upload or record your singing voice.

    - Reference: Upload or record the original song/audio.

    - Action: Click "🚀 Start Analysis".

## 🧠 How It Works

* **Feature Extraction**: The system extracts **Chroma CQT** (for pitch classes) and **RMS** (for energy/rhythm) features using `librosa`.
* **Alignment**: It calculates the optimal path between the two audio sequences using **Dynamic Time Warping (DTW)**. This ensures that even if you sing slightly faster or slower, the system compares the correct musical moments.
* **Scoring Logic**:
    * **Pitch Score**: Calculated using Cosine Similarity between normalized chroma vectors.
    * **Tempo Score**: Analyzes the onset strength envelopes to compare rhythmic patterns.
    * **Overall Score**: A weighted average (70% Pitch, 30% Tempo).
* **Visualization**:
    * `Matplotlib` creates the static graphs.
    * `Matplotlib.animation` + `FFmpeg` renders the frame-by-frame pitch tracking video overlay.

## ⚠️ Known Limitations

* **Computation Time**: Generating the animation video is computationally intensive and may take 2-3x the duration of the audio file. You can uncheck "Generate animation video" for faster results.
* **Audio Length**: Short clips (< 3 seconds) may not have enough data for the windowed analysis.
* **FFmpeg**: If the video output is empty, ensure `ffmpeg` is correctly installed and accessible via your terminal.