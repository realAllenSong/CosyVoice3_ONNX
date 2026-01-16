"""
Configuration management for CosyVoice3 ONNX.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


def get_default_model_dir() -> Path:
    """Get the default model directory."""
    return Path.home() / ".cosyvoice3" / "models"


def get_default_config_dir() -> Path:
    """Get the default config directory."""
    return Path.home() / ".cosyvoice3"


@dataclass
class CosyVoiceConfig:
    """Configuration for CosyVoice3 ONNX."""
    
    # Model settings
    model_dir: str = field(default_factory=lambda: str(get_default_model_dir()))
    precision: str = "auto"  # "fp16", "fp32", or "auto"
    num_threads: int = 0  # 0 = auto
    
    # Audio settings
    sample_rate: int = 24000
    default_speed: float = 1.0
    default_volume: float = 1.0
    
    # Generation settings
    sampling_k: int = 25
    max_tokens: int = 500
    min_tokens: int = 10
    n_timesteps: int = 10
    
    # Resource settings
    preload: bool = False
    lazy_load: bool = True
    
    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: Optional[str] = None
    
    # Network
    auto_download: bool = True
    hf_repo_id: str = "ayousanz/cosy-voice3-onnx"
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "CosyVoiceConfig":
        """Load configuration from file."""
        if config_path is None:
            config_path = get_default_config_dir() / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        
        return cls()
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        if config_path is None:
            config_path = get_default_config_dir() / "config.yaml"
        else:
            config_path = Path(config_path)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
    
    def get_model_path(self, filename: str) -> Path:
        """Get full path for a model file."""
        return Path(self.model_dir) / filename
    
    def get_precision_suffix(self) -> str:
        """Get the precision suffix for model files."""
        if self.precision == "auto":
            # Auto-detect based on ONNX Runtime version
            try:
                import onnxruntime as ort
                version = tuple(map(int, ort.__version__.split('.')[:2]))
                # FP16 requires ONNX Runtime >= 1.18
                if version >= (1, 18):
                    return "_fp16"
                else:
                    return "_fp32"
            except Exception:
                return "_fp32"
        elif self.precision == "fp16":
            return "_fp16"
        else:
            return "_fp32"
