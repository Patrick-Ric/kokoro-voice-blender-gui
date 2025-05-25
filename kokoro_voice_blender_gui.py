import sys
import json
import os
import random
import numpy as np
try:
    import kokoro_onnx
except ImportError as e:
    print(f"Error importing kokoro_onnx: {e}")
    print("Please ensure 'kokoro-onnx' is installed in the active Python environment.")
    print("Run: pip install kokoro-onnx")
    sys.exit(1)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QSlider, QMessageBox, QScrollArea, QSplitter, QCheckBox,
    QComboBox, QGridLayout, QSpacerItem, QFileDialog, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMouseEvent
import soundfile as sf
import pygame

class CustomSlider(QSlider):
    """Custom QSlider that jumps to the clicked position."""
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Calculate the clicked position
            val = self.minimum() + ((self.maximum() - self.minimum()) * event.x()) // self.width()
            self.setValue(int(val))
        super().mousePressEvent(event)

class KokoroVoiceBlender(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kokoro Voice Blender")
        self.setGeometry(100, 100, 800, 700)

        # Paths to model files
        self.model_path = "/home/pg/Dokumente/Kokoro-82M/kokoro.onnx"
        self.voices_path = "/home/pg/Dokumente/Kokoro-82M/voices-v1.0.bin"
        self.config_dir = "/home/pg/Dokumente/Kokoro-82M/configs"
        self.last_config_path = os.path.join(self.config_dir, "last_blender_config.json")

        # Initialize Kokoro pipeline (CPU only)
        self.device = "cpu"
        try:
            self.pipeline = kokoro_onnx.Kokoro(
                model_path=self.model_path,
                voices_path=self.voices_path
            )
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to initialize Kokoro pipeline: {str(e)}")
            sys.exit(1)

        # Available voices
        self.voices = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "ef_dora", "em_alex", "em_santa",
            "ff_siwis",
            "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            "if_sara", "im_nicola",
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
            "pf_dora", "pm_alex", "pm_santa",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxia", "zm_yunxi", "zm_yunyang"
        ]

        # Initialize sliders and labels
        self.sliders = {}
        self.labels = {}
        self.slider_changed = False
        self.columns = 1  # Default: 1 slider per row
        self.normalize_sliders = True  # Default: Normalize sliders to sum to 1
        self.speed = 1.0  # Default: Normal speed
        self.adjusting = False  # Lock to prevent recursive updates

        # Debouncing für Slider-Signale
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.process_debounced_slider_change)
        self.pending_voice = None
        self.pending_value = None

        # Auto-loop variables
        self.auto_loop = False
        self.continuous_loop = False
        self.loop_timer = QTimer()
        self.loop_timer.timeout.connect(self.run_auto_loop)

        # Setup GUI
        self.init_ui()

        # Load last configuration if exists
        self.load_last_config()

    def init_ui(self):
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Splitter for resizable sections
        splitter = QSplitter(Qt.Vertical)
        main_widget.setLayout(QVBoxLayout())
        main_widget.layout().addWidget(splitter)

        # Section 1: Text input
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.addWidget(QLabel("Text to Synthesize:"))
        self.text_input = QTextEdit()
        self.text_input.setText("Hello, this is a test for voice blending.")
        text_layout.addWidget(self.text_input)
        splitter.addWidget(text_widget)

        # Section 2: Sliders in scroll area
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.scroll_widget)
        self.update_slider_layout()
        splitter.addWidget(scroll_area)

        # Section 3: Buttons and checkboxes
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        
        # Checkboxes and controls
        controls_layout = QHBoxLayout()
        self.auto_loop_cb = QCheckBox("Auto-Loop Preview")
        self.auto_loop_cb.stateChanged.connect(self.toggle_auto_loop)
        controls_layout.addWidget(self.auto_loop_cb)
        
        self.continuous_loop_cb = QCheckBox("Continuous Loop")
        self.continuous_loop_cb.setEnabled(False)
        self.continuous_loop_cb.stateChanged.connect(self.toggle_continuous_loop)
        controls_layout.addWidget(self.continuous_loop_cb)
        
        self.normalize_cb = QCheckBox("Normalize Sliders to Sum 1")
        self.normalize_cb.setChecked(True)
        self.normalize_cb.stateChanged.connect(self.toggle_normalize_sliders)
        controls_layout.addWidget(self.normalize_cb)
        
        controls_layout.addSpacerItem(QSpacerItem(20, 20))
        controls_layout.addWidget(QLabel("Sliders per Row:"))
        self.columns_combo = QComboBox()
        self.columns_combo.addItems(["1", "2", "3", "4", "5"])
        self.columns_combo.currentIndexChanged.connect(self.change_columns)
        controls_layout.addWidget(self.columns_combo)
        
        controls_layout.addWidget(QLabel("Random Voices:"))
        self.random_voice_count_combo = QComboBox()
        self.random_voice_count_combo.addItems([str(i) for i in range(1, 21)])
        self.random_voice_count_combo.setCurrentText("10")
        controls_layout.addWidget(self.random_voice_count_combo)
        
        self.randomize_btn = QPushButton("Randomize")
        self.randomize_btn.clicked.connect(self.randomize_voices)
        controls_layout.addWidget(self.randomize_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_voices)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.1, 3.0)
        self.speed_spinbox.setValue(1.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.valueChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_spinbox)
        
        controls_layout.addStretch()
        button_layout.addLayout(controls_layout)

        # Additional buttons
        extra_buttons_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Sliders")
        self.reset_btn.clicked.connect(self.reset_sliders)
        extra_buttons_layout.addWidget(self.reset_btn)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        extra_buttons_layout.addWidget(self.save_config_btn)

        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_config)
        extra_buttons_layout.addWidget(self.load_config_btn)
        extra_buttons_layout.addStretch()
        button_layout.addLayout(extra_buttons_layout)

        # Main buttons
        buttons_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Blend")
        self.preview_btn.clicked.connect(self.preview_blend)
        buttons_layout.addWidget(self.preview_btn)

        self.synthesize_btn = QPushButton("Synthesize and Save")
        self.synthesize_btn.clicked.connect(self.synthesize_and_save)
        buttons_layout.addWidget(self.synthesize_btn)
        button_layout.addLayout(buttons_layout)
        splitter.addWidget(button_widget)

        # Set initial sizes
        splitter.setSizes([100, 400, 100])

    def update_speed(self, value):
        self.speed = value

    def toggle_normalize_sliders(self, state):
        self.normalize_sliders = state == Qt.Checked
        if self.normalize_sliders:
            self.adjust_sliders_to_sum_one(None)
        self.update_labels()

    def slider_value_changed(self, voice, value):
        if not self.adjusting:
            self.pending_voice = voice
            self.pending_value = value
            self.debounce_timer.start(100)  # 100 ms Verzögerung

    def process_debounced_slider_change(self):
        if self.pending_voice is not None:
            if self.normalize_sliders:
                self.adjust_sliders_to_sum_one(self.pending_voice)
            else:
                self.update_labels()
            self.pending_voice = None
            self.pending_value = None

    def adjust_sliders_to_sum_one(self, changed_voice):
        if not self.normalize_sliders or self.adjusting:
            return
        
        self.adjusting = True
        
        # Get current normalized values (0-1)
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}
        total = sum(voice_ratios.values())
        
        if total == 0:
            self.adjusting = False
            return
        
        # Store new values to apply later
        new_values = {}
        
        if changed_voice:
            # Get the new value of the changed slider
            target_value = voice_ratios[changed_voice]
            
            # Identify active voices (value > 0, excluding changed voice)
            active_voices = [(voice, value) for voice, value in voice_ratios.items() if voice != changed_voice and value > 0]
            remaining_sum = sum(value for _, value in active_voices)
            
            if remaining_sum == 0:
                for voice in self.voices:
                    if voice != changed_voice:
                        new_values[voice] = 0
                new_values[changed_voice] = min(1.0, target_value)
            else:
                # Calculate the remaining weight to distribute
                remaining_weight = 1 - target_value
                
                if remaining_weight < 0:
                    new_values[changed_voice] = 1.0
                    for voice in self.voices:
                        if voice != changed_voice:
                            new_values[voice] = 0
                else:
                    # Store new values proportionally
                    new_values[changed_voice] = target_value
                    for voice, current_value in active_voices:
                        proportion = current_value / remaining_sum
                        new_value = remaining_weight * proportion
                        new_values[voice] = new_value
                    
                    # Adjust the last active voice to ensure sum is exactly 1
                    active_voices_keys = [v for v, _ in active_voices]
                    if active_voices_keys:
                        last_voice = active_voices_keys[-1]
                        current_sum = sum(new_values.get(v, 0) for v in self.voices)
                        adjustment = 1 - current_sum
                        new_values[last_voice] = new_values[last_voice] + adjustment
        else:
            # Scale all voices to sum to 1
            scale_factor = 1 / total if total > 0 else 0
            for voice in self.voices:
                new_value = voice_ratios[voice] * scale_factor
                new_values[voice] = new_value
            
            # Adjust the last non-zero voice to ensure sum is exactly 1
            non_zero_voices = [v for v, r in voice_ratios.items() if r > 0]
            if non_zero_voices:
                last_voice = non_zero_voices[-1]
                current_sum = sum(new_values.get(v, 0) for v in self.voices)
                adjustment = 1 - current_sum
                new_values[last_voice] = new_values[last_voice] + adjustment
        
        # Apply new values with signals blocked
        for voice in self.voices:
            new_value = new_values.get(voice, 0)
            slider_value = max(0, min(100, round(new_value * 100)))
            self.sliders[voice].blockSignals(True)
            self.sliders[voice].setValue(slider_value)
            self.sliders[voice].blockSignals(False)
        
        self.adjusting = False
        self.update_labels()

    def update_slider_layout(self):
        # Save current slider values
        current_values = {voice: slider.value() for voice, slider in self.sliders.items()}

        # Clear existing layout
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            if item.layout():
                layout = item.layout()
                for j in reversed(range(layout.count())):
                    widget = layout.itemAt(j).widget()
                    if widget:
                        widget.setParent(None)
                layout.setParent(None)
            elif item.widget():
                item.widget().setParent(None)

        # Reset sliders and labels dictionaries
        self.sliders = {}
        self.labels = {}

        # Add sliders in grid layout
        for idx, voice in enumerate(self.voices):
            row = idx // self.columns
            col = idx % self.columns
            slider_layout = QHBoxLayout()
            label = QLabel(f"{voice}: 0.0")
            slider_layout.addWidget(label)
            slider = CustomSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(current_values.get(voice, 0))  # Restore saved value
            slider.setTracking(True)
            slider.setSingleStep(1)
            slider.valueChanged.connect(lambda value, v=voice: self.slider_value_changed(v, value))
            slider_layout.addWidget(slider)
            self.sliders[voice] = slider
            self.labels[voice] = label
            self.scroll_layout.addLayout(slider_layout, row, col)

        if self.normalize_sliders:
            self.adjust_sliders_to_sum_one(None)
        self.update_labels()

    def change_columns(self):
        self.columns = int(self.columns_combo.currentText())
        self.update_slider_layout()

    def reset_sliders(self):
        for slider in self.sliders.values():
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.update_labels()

    def randomize_voices(self):
        # Get number of voices to randomize
        num_voices = int(self.random_voice_count_combo.currentText())
        num_voices = min(num_voices, len(self.voices))  # Ensure not more than available voices

        # Randomly select voices
        selected_voices = random.sample(self.voices, num_voices)

        # Generate random weights
        if self.normalize_sliders:
            # Use Dirichlet distribution for weights summing to 1
            weights = np.random.dirichlet(np.ones(num_voices))
        else:
            # Generate raw weights between 0.01 and 1.00
            weights = np.random.uniform(0.01, 1.00, num_voices)

        # Assign weights to sliders
        for voice in self.voices:
            self.sliders[voice].blockSignals(True)
            if voice in selected_voices:
                idx = selected_voices.index(voice)
                slider_value = max(1, min(100, round(weights[idx] * 100)))
                self.sliders[voice].setValue(slider_value)
            else:
                self.sliders[voice].setValue(0)
            self.sliders[voice].blockSignals(False)

        if self.normalize_sliders:
            self.adjust_sliders_to_sum_one(None)
        self.update_labels()

    def refresh_voices(self):
        # Get currently active voices (value > 0)
        active_voices = [voice for voice, slider in self.sliders.items() if slider.value() > 0]
        if not active_voices:
            QMessageBox.warning(self, "Warning", "No active voices to refresh. Please randomize or set voices first.")
            return

        # Generate new random weights for active voices
        num_voices = len(active_voices)
        if self.normalize_sliders:
            # Use Dirichlet distribution for weights summing to 1
            weights = np.random.dirichlet(np.ones(num_voices))
        else:
            # Generate raw weights between 0.01 and 1.00
            weights = np.random.uniform(0.01, 1.00, num_voices)

        # Assign new weights to active voices
        for voice in self.voices:
            self.sliders[voice].blockSignals(True)
            if voice in active_voices:
                idx = active_voices.index(voice)
                slider_value = max(1, min(100, round(weights[idx] * 100)))
                self.sliders[voice].setValue(slider_value)
            self.sliders[voice].blockSignals(False)

        if self.normalize_sliders:
            self.adjust_sliders_to_sum_one(None)
        self.update_labels()

        # Play the new blend
        self.preview_blend()

    def save_config(self):
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)

        # Get normalized weights
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}

        # Prepare config
        config = {
            "voice_weights": voice_ratios,
            "voice_enabled": {voice: slider.value() > 0 for voice, slider in self.sliders.items()},
            "normalize_sliders": self.normalize_sliders,
            "sliders_per_row": self.columns,
            "speed": self.speed
        }

        # Open file dialog with automatic .json suffix
        file_dialog = QFileDialog(self, "Save Configuration", self.config_dir, "JSON Files (*.json)")
        file_dialog.setDefaultSuffix("json")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_config(self):
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", self.config_dir, "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                voice_weights = config.get("voice_weights", {})
                for voice in self.voices:
                    if voice in voice_weights:
                        # Scale normalized weights (0-1) to slider range (0-100)
                        self.sliders[voice].blockSignals(True)
                        self.sliders[voice].setValue(int(voice_weights[voice] * 100))
                        self.sliders[voice].blockSignals(False)
                    else:
                        self.sliders[voice].blockSignals(True)
                        self.sliders[voice].setValue(0)
                        self.sliders[voice].blockSignals(False)
                
                # Load normalize_sliders setting
                self.normalize_sliders = config.get("normalize_sliders", True)
                self.normalize_cb.setChecked(self.normalize_sliders)
                
                # Load sliders_per_row setting
                self.columns = config.get("sliders_per_row", 1)
                self.columns_combo.setCurrentText(str(self.columns))
                
                # Load speed setting
                self.speed = config.get("speed", 1.0)
                self.speed_spinbox.setValue(self.speed)
                
                if self.normalize_sliders:
                    self.adjust_sliders_to_sum_one(None)
                self.update_slider_layout()
                self.update_labels()
                QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")

    def load_last_config(self):
        if os.path.exists(self.last_config_path):
            try:
                with open(self.last_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                voice_weights = config.get("voice_weights", {})
                for voice in self.voices:
                    if voice in voice_weights:
                        self.sliders[voice].blockSignals(True)
                        self.sliders[voice].setValue(int(voice_weights[voice] * 100))
                        self.sliders[voice].blockSignals(False)
                    else:
                        self.sliders[voice].blockSignals(True)
                        self.sliders[voice].setValue(0)
                        self.sliders[voice].blockSignals(False)
                
                # Load normalize_sliders setting
                self.normalize_sliders = config.get("normalize_sliders", True)
                self.normalize_cb.setChecked(self.normalize_sliders)
                
                # Load sliders_per_row setting
                self.columns = config.get("sliders_per_row", 1)
                self.columns_combo.setCurrentText(str(self.columns))
                
                # Load speed setting
                self.speed = config.get("speed", 1.0)
                self.speed_spinbox.setValue(self.speed)
                
                if self.normalize_sliders:
                    self.adjust_sliders_to_sum_one(None)
                self.update_slider_layout()
                self.update_labels()
            except Exception as e:
                print(f"Failed to load last configuration: {str(e)}")

    def closeEvent(self, event):
        # Save current configuration as last_blender_config.json
        os.makedirs(self.config_dir, exist_ok=True)
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}
        config = {
            "voice_weights": voice_ratios,
            "voice_enabled": {voice: slider.value() > 0 for voice, slider in self.sliders.items()},
            "normalize_sliders": self.normalize_sliders,
            "sliders_per_row": self.columns,
            "speed": self.speed
        }
        try:
            with open(self.last_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save last configuration: {str(e)}")
        
        super().closeEvent(event)

    def update_labels(self):
        # Get normalized values (0-1)
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}
        
        # Update labels with normalized values
        for voice, slider in self.sliders.items():
            norm_value = slider.value() / 100
            self.labels[voice].setText(f"{voice}: {norm_value:.2f}")

        # Mark sliders as changed for auto-loop
        self.slider_changed = True

    def toggle_auto_loop(self, state):
        self.auto_loop = state == Qt.Checked
        self.continuous_loop_cb.setEnabled(self.auto_loop)
        if self.auto_loop:
            self.slider_changed = True
            self.loop_timer.start(2000)  # 2 seconds for full playback
        else:
            self.continuous_loop_cb.setChecked(False)
            self.continuous_loop = False
            self.loop_timer.stop()
            pygame.mixer.quit()

    def toggle_continuous_loop(self, state):
        self.continuous_loop = state == Qt.Checked

    def run_auto_loop(self):
        if not self.auto_loop or (not self.continuous_loop and not self.slider_changed):
            return

        # Wait for current playback to finish
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            return

        # Run preview blend
        self.preview_blend(auto_loop=True)
        self.slider_changed = False

    def preview_blend(self, auto_loop=False):
        text = self.text_input.toPlainText().strip()
        if not text:
            if not auto_loop:
                QMessageBox.critical(self, "Error", "Please enter text to synthesize.")
            return

        # Get normalized values (0-1)
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}
        active_voices = [(voice, ratio) for voice, ratio in voice_ratios.items() if ratio > 0]

        if not active_voices:
            if not auto_loop:
                QMessageBox.critical(self, "Error", "At least one voice ratio must be greater than 0.")
            return

        # Calculate total sum of ratios
        total = sum(ratio for _, ratio in active_voices)
        if total == 0:
            total = 1  # Prevent division by zero

        # Scale ratios if normalization is disabled
        scaled_voices = [(voice, ratio / total if not self.normalize_sliders else ratio) for voice, ratio in active_voices]

        # Create voice blending
        try:
            voice_blend = sum(
                self.pipeline.voices[voice] * ratio
                for voice, ratio in scaled_voices
            )
        except KeyError as e:
            if not auto_loop:
                QMessageBox.critical(self, "Error", f"Voice not found: {str(e)}")
            return

        temp_file = "temp_preview.wav"

        try:
            # Synthesize audio
            samples, sr = self.pipeline.create(text, voice=voice_blend, speed=self.speed, lang="en-us")
            sf.write(temp_file, samples, sr)

            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            if not auto_loop:
                while pygame.mixer.music.get_busy():
                    QApplication.processEvents()
        except Exception as e:
            if not auto_loop:
                QMessageBox.critical(self, "Error", f"Failed to preview: {str(e)}")

    def synthesize_and_save(self):
        # Stop auto-loop if running
        if self.auto_loop:
            self.auto_loop_cb.setChecked(False)

        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.critical(self, "Error", "Please enter text to synthesize.")
            return

        # Get normalized values (0-1)
        voice_ratios = {voice: slider.value() / 100 for voice, slider in self.sliders.items()}
        active_voices = [(voice, ratio) for voice, ratio in voice_ratios.items() if ratio > 0]

        if not active_voices:
            QMessageBox.critical(self, "Error", "At least one voice ratio must be greater than 0.")
            return

        # Calculate total sum of ratios
        total = sum(ratio for _, ratio in active_voices)
        if total == 0:
            total = 1  # Prevent division by zero

        # Scale ratios if normalization is disabled
        scaled_voices = [(voice, ratio / total if not self.normalize_sliders else ratio) for voice, ratio in active_voices]

        # Create voice blending
        try:
            voice_blend = sum(
                self.pipeline.voices[voice] * ratio
                for voice, ratio in scaled_voices
            )
        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Voice not found: {str(e)}")
            return

        output_file = "output_blended.wav"

        try:
            # Synthesize audio
            samples, sr = self.pipeline.create(text, voice=voice_blend, speed=self.speed, lang="en-us")
            sf.write(output_file, samples, sr)

            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                QApplication.processEvents()

            QMessageBox.information(self, "Success", f"Audio saved as {output_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to synthesize: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KokoroVoiceBlender()
    window.show()
    sys.exit(app.exec_())