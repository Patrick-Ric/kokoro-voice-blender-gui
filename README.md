# Kokoro Voice Blender GUI

A PyQt5-based graphical user interface (GUI) for blending multiple voices using the Kokoro ONNX text-to-speech (TTS) model. This tool allows users to mix various voices with customizable weights, adjust playback speed, and save configurations for later use. It complements the [Kokoro TTS GUI](https://github.com/Patrick-Ric/kokoro-tts-gui) by sharing the same configuration directory (`/home/pg/Dokumente/Kokoro-82M/configs/` by default), enabling seamless voice mix exchanges between the two applications.

## Features

The Kokoro Voice Blender GUI offers a rich set of features for voice blending and audio synthesis:

### 1. Text-to-Speech Input
- **Text Input**: Enter any text to be synthesized into speech via a text area.

### 2. Voice Blending with Sliders
- **Multiple Voices**: Choose from a predefined list of 50+ voices (e.g., `af_alloy`, `im_nicola`, `zf_xiaoyi`).
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
- **Save Config**: Save voice weights, normalization settings, slider layout, and speed to a JSON file in the `configs/` directory (default: `/home/pg/Dokumente/Kokoro-82M/configs/`).
- **Load Config**: Load previously saved configurations.
- **Last Config**: Automatically saves the current state on exit and loads it on startup.
- **Shared Configs**: Uses the same `configs/` directory as [Kokoro TTS GUI](https://github.com/Patrick-Ric/kokoro-tts-gui) for interoperability.

### 6. Customization Options
- **Sliders per Row**: Adjust the GUI layout (1 to 5 sliders per row) for better usability.
- **Speed Control**: Modify playback speed (0.1x to 3.0x) using a spin box.
- **Reset Sliders**: Set all sliders to 0.00 to start fresh.

## Installation

### Prerequisites
- Python 3.8 or higher
- Kokoro ONNX model files (`kokoro.onnx`, `voices-v1.0.bin`) in a directory accessible to the application (e.g., `/home/pg/Dokumente/Kokoro-82M/`)
- A compatible audio backend (e.g., `pygame` for playback)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Patrick-Ric/kokoro-voice-blender-gui.git
   cd kokoro-voice-blender-gui
2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
    pip install -r requirements.txt
4. Ensure the Kokoro model files (kokoro.onnx, voices-v1.0.bin) are in the expected directory (default: /home/pg/Dokumente/Kokoro-82M/).
    Update the paths in kokoro_voice_blender_gui.py (model_path, voices_path, config_dir) if your setup differs.
5. Run the application:
    python kokoro_voice_blender_gui.py
   
   
