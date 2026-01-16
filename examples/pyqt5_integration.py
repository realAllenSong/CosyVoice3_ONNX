#!/usr/bin/env python3
"""
PyQt5 Integration Example for CosyVoice3 ONNX

This example shows how to integrate CosyVoice TTS with a PyQt5 application,
using signals for async operation.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QProgressBar
)

from cosyvoice_onnx import CosyVoiceTTS


class TTSWorker(QThread):
    """Worker thread for TTS synthesis."""
    
    started = pyqtSignal()
    progress = pyqtSignal(float)
    finished = pyqtSignal(bytes, int)  # audio data, sample_rate
    error = pyqtSignal(str)
    
    def __init__(self, tts: CosyVoiceTTS, text: str, prompt_audio: str, prompt_text: str):
        super().__init__()
        self.tts = tts
        self.text = text
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self._stop = False
    
    def run(self):
        self.started.emit()
        try:
            def on_progress(info):
                if not self._stop:
                    self.progress.emit(info.progress_percent)
            
            audio = self.tts.clone_voice(
                prompt_audio=self.prompt_audio,
                prompt_text=self.prompt_text,
                target_text=self.text
            )
            
            if not self._stop:
                self.finished.emit(audio.to_bytes(), audio.sample_rate)
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._stop = True


class TTSWidget(QWidget):
    """Main TTS control widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tts = None
        self.worker = None
        self.prompt_audio = None
        self.prompt_text = ""
        
        self.setup_ui()
        self.setup_tts()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("CosyVoice3 ONNX - PyQt5 Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # Reference voice section
        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference: None")
        ref_btn = QPushButton("Load Reference Audio")
        ref_btn.clicked.connect(self.load_reference)
        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(ref_btn)
        layout.addLayout(ref_layout)
        
        # Reference transcript
        layout.addWidget(QLabel("Reference Transcript:"))
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setMaximumHeight(60)
        self.transcript_edit.setPlaceholderText("Enter the transcript of your reference audio...")
        layout.addWidget(self.transcript_edit)
        
        # Text to synthesize
        layout.addWidget(QLabel("Text to Synthesize:"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text to synthesize...")
        layout.addWidget(self.text_edit)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.synth_btn = QPushButton("🎙️ Synthesize")
        self.synth_btn.clicked.connect(self.synthesize)
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_synthesis)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.synth_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        
        # Status
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
    
    def setup_tts(self):
        """Initialize TTS engine."""
        self.status.setText("Loading TTS engine...")
        try:
            self.tts = CosyVoiceTTS(log_level="WARNING")
            self.status.setText("Ready - TTS engine loaded")
        except Exception as e:
            self.status.setText(f"Error: {e}")
    
    def load_reference(self):
        """Load reference audio file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio",
            "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if path:
            self.prompt_audio = path
            self.ref_label.setText(f"Reference: {Path(path).name}")
    
    def synthesize(self):
        """Start synthesis."""
        if self.tts is None:
            self.status.setText("TTS engine not loaded")
            return
        
        if not self.prompt_audio:
            self.status.setText("Please load reference audio first")
            return
        
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.status.setText("Please enter text to synthesize")
            return
        
        prompt_text = self.transcript_edit.toPlainText().strip()
        if not prompt_text:
            self.status.setText("Please enter reference transcript")
            return
        
        # Start worker
        self.worker = TTSWorker(self.tts, text, self.prompt_audio, prompt_text)
        self.worker.started.connect(self.on_started)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def stop_synthesis(self):
        """Stop ongoing synthesis."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.status.setText("Stopped")
            self.synth_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def on_started(self):
        self.synth_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("Synthesizing...")
    
    def on_progress(self, percent):
        self.progress.setValue(int(percent))
    
    def on_finished(self, audio_data, sample_rate):
        self.synth_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        self.status.setText(f"Done! Audio: {len(audio_data)} bytes @ {sample_rate}Hz")
        
        # Save dialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "output.wav", "WAV Files (*.wav)"
        )
        if path:
            with open(path, 'wb') as f:
                f.write(audio_data)
            self.status.setText(f"Saved to: {path}")
    
    def on_error(self, error):
        self.synth_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"Error: {error}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CosyVoice3 ONNX - PyQt5 Demo")
        self.setMinimumSize(500, 400)
        
        self.widget = TTSWidget(self)
        self.setCentralWidget(self.widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
