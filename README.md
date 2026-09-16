# Kokoro Voice Blender GUI v1.1.0

A PyQt5-based graphical user interface (GUI) for blending multiple voices using the Kokoro ONNX text-to-speech (TTS) model. This tool allows users to mix various voices with customizable weights, adjust playback speed, and save configurations for later use. It complements the [Kokoro TTS GUI](https://github.com/Patrick-Ric/kokoro-tts-gui): place both scripts in the same directory and they share the `configs/` subdirectory, enabling seamless voice mix exchanges between the two applications.

## Features

The Kokoro Voice Blender GUI offers a rich set of features for voice blending and audio synthesis:

### 1. Text-to-Speech Input
- **Text Input**: Enter any text to be synthesized into speech via a text area.

### 2. Voice Blending with Sliders
- **All Voices**: The voice list is read from `voices-v1.0.bin`, so it always matches the voices file (e.g., `af_heart`, `im_nicola`, `zf_xiaoyi`).
- **Customizable Weights**: Adjust the contribution of each voice using sliders (range: 0.00 to 1.00).
- **Normalization Option**:
  - **Enabled**: Automatically adjusts slider values to sum to 1.00, ensuring balanced blending.
  - **Disabled**: Allows raw weights, with internal scaling for audio output to maintain full intensity.
- **Responsive Sliders**: Click anywhere on a slider to jump to a specific value.

### 3. Random Voice Mixing
- **Randomize Button**: Select 1 to 20 voices and assign random weights.
  - With normalization: Weights sum to 1.00 (using Dirichlet distribution).
  - Without normalization: Weights range from 0.01 to 1.00.
- **Refresh Button**: Re-randomizes weights for currently active voices and plays the new blend immediately.

### 4. Audio Playback and Saving
- **Preview Blend**: Synthesize and play the blended voice mix in real-time.
- **Synthesize and Save**: Save the synthesized audio as `output_blended.wav`.
- **Auto-Loop Preview**:
  - Automatically replays the blend after changes or continuously if enabled.
  - Controlled via "Auto-Loop Preview" and "Continuous Loop" checkboxes.

### 5. Configuration Management
- **Save Config**: Save voice weights, normalization settings, slider layout, and speed to a JSON file in the `configs/` directory next to the script.
- **Load Config**: Load previously saved configurations.
- **Last Config**: Automatically saves the current state on exit and loads it on startup.
- **Shared Configs**: Uses the same `configs/` directory as [Kokoro TTS GUI](https://github.com/Patrick-Ric/kokoro-tts-gui) for interoperability (when both scripts are in the same folder).

### 6. Customization Options
- **Sliders per Row**: Adjust the GUI layout (1 to 5 sliders per row) for better usability.
- **Speed Control**: Modify playback speed (0.5x to 2.0x, the range supported by kokoro-onnx) using a spin box.
- **Reset Sliders**: Set all sliders to 0.00 to start fresh.
- **CPU by default**: runs on any PC without a GPU (tested). NVIDIA GPU acceleration via `pip install "kokoro-onnx[gpu]"` should work automatically but is untested — feedback welcome.

## Screenshot
![Voice Blender GUI](https://github.com/user-attachments/assets/7bcb3f72-a976-49b3-ad6c-22c686007a8e)

## Installation

### Prerequisites
- Python 3.10–3.13 (required: `kokoro-onnx>=0.4.7` needs Python ≥ 3.10; with Python 3.9 pip silently installs an old `kokoro-onnx 0.1.x` that cannot read `voices-v1.0.bin`)
- Kokoro ONNX model files next to the script (**no renaming needed**: `kokoro-v1.0.onnx` as shipped upstream works, as do `kokoro.onnx` or any `kokoro*.onnx`; same for `voices*.bin`).
  Download: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
- A compatible audio backend (e.g., `pygame` for playback)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Patrick-Ric/kokoro-voice-blender-gui.git
   cd kokoro-voice-blender-gui
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies (CPU-only — no NVIDIA/CUDA downloads):
   ```bash
   pip install -r requirements.txt
   ```
   Optional, only for NVIDIA GPUs (adds ~1 GB CUDA libraries):
   ```bash
   pip install "kokoro-onnx[gpu]"
   ```
4. Place the model file (`kokoro-v1.0.onnx`) and `voices-v1.0.bin` next to the script. Tip: also place `kokoro_tts_gui.py` from the [Kokoro TTS GUI](https://github.com/Patrick-Ric/kokoro-tts-gui) in the same directory so both programs share the `configs/` subdirectory.
5. Run the application:
   ```bash
   python kokoro_voice_blender_gui.py
   ```
