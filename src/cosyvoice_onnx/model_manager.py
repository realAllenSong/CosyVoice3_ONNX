"""
Model manager for CosyVoice3 ONNX.
Handles model downloading, loading, and lifecycle management.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import threading

import numpy as np
import onnxruntime as ort

from .config import CosyVoiceConfig, get_default_model_dir
from .utils.logger import get_logger


class ModelNotFoundError(Exception):
    """Raised when required model files are not found."""
    pass


class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelManager:
    """Manages ONNX model loading and lifecycle.
    
    Handles:
    - Auto-downloading models from HuggingFace
    - Loading ONNX sessions with appropriate providers
    - Lazy loading and preloading
    - Memory management
    """
    
    # Required model files for CosyVoice3
    REQUIRED_MODELS = [
        # LLM models
        "llm_backbone_initial_{precision}.onnx",
        "llm_backbone_decode_{precision}.onnx",
        "llm_decoder_{precision}.onnx",
        "llm_speech_embedding_{precision}.onnx",
        # Flow models  
        "flow_token_embedding_{precision}.onnx",
        "flow_speaker_projection_{precision}.onnx",
        "flow_pre_lookahead_{precision}.onnx",
        "flow.decoder.estimator.{precision}.onnx",
        # HiFT models (always FP32 for stability)
        "hift_f0_predictor_fp32.onnx",
        "hift_source_generator_fp32.onnx",
        "hift_decoder_fp32.onnx",
        # Audio processing
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        # Text processing
        "text_embedding_fp32.onnx",
    ]
    
    # Tokenizer files
    TOKENIZER_FILES = [
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ]
    
    def __init__(self, config: Optional[CosyVoiceConfig] = None):
        """Initialize the model manager.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or CosyVoiceConfig()
        self.logger = get_logger()
        
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._tokenizer = None
        self._loaded = False
        self._lock = threading.Lock()
        
        # ONNX Runtime settings - match ayousanz reference script
        self._session_options = ort.SessionOptions()
        self._session_options.log_severity_level = 3  # ERROR only
        
        if self.config.num_threads > 0:
            self._session_options.intra_op_num_threads = self.config.num_threads
            self._session_options.inter_op_num_threads = self.config.num_threads
        
        # Detect providers
        self._providers = self._detect_providers()
        
    def _detect_providers(self) -> List[str]:
        """Detect available ONNX Runtime providers."""
        available = ort.get_available_providers()
        providers = []
        
        if 'CUDAExecutionProvider' in available:
            # Test if CUDA actually works
            try:
                test_path = os.path.join(self.config.model_dir, "text_embedding_fp32.onnx")
                if os.path.exists(test_path):
                    _ = ort.InferenceSession(
                        test_path,
                        self._session_options,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    providers.append('CUDAExecutionProvider')
                    self.logger.info("CUDA GPU acceleration available")
            except Exception as e:
                self.logger.debug(f"CUDA not usable: {e}")
        
        # Note: CoreML is disabled due to compatibility issues with FP16 models
        # Apple Silicon still benefits from optimized CPU operations
        
        providers.append('CPUExecutionProvider')
        self.logger.info(f"Using providers: {providers}")
        return providers
    
    def ensure_models_exist(
        self, 
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """Ensure all required models are downloaded.
        
        Args:
            on_progress: Callback for download progress (filename, percent)
            
        Returns:
            True if all models exist, False if download failed
            
        Raises:
            ModelDownloadError: If auto_download is True but download fails
        """
        model_dir = Path(self.config.model_dir)
        precision = self.config.get_precision_suffix().replace("_", "")
        
        # Check which files are missing
        missing = []
        for template in self.REQUIRED_MODELS:
            filename = template.format(precision=precision)
            if not (model_dir / filename).exists():
                # Try alternate precision
                alt_precision = "fp32" if precision == "fp16" else "fp16"
                alt_filename = template.format(precision=alt_precision)
                if not (model_dir / alt_filename).exists():
                    missing.append(filename)
        
        for filename in self.TOKENIZER_FILES:
            if not (model_dir / filename).exists():
                missing.append(filename)
        
        if not missing:
            self.logger.info("All required models found")
            return True
        
        self.logger.info(f"Missing {len(missing)} files, downloading...")
        
        if not self.config.auto_download:
            raise ModelNotFoundError(
                f"Missing model files: {missing}. "
                f"Set auto_download=True or manually download from {self.config.hf_repo_id}"
            )
        
        # Download from HuggingFace
        return self._download_models(on_progress)
    
    def _download_models(
        self, 
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """Download models from HuggingFace.
        
        Args:
            on_progress: Callback for download progress
            
        Returns:
            True if download successful
        """
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            
            model_dir = Path(self.config.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Downloading models from {self.config.hf_repo_id}...")
            
            # Download entire repository
            snapshot_download(
                repo_id=self.config.hf_repo_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            
            self.logger.info("Model download complete!")
            
            if on_progress:
                on_progress("complete", 100.0)
            
            return True
            
        except ImportError:
            raise ModelDownloadError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
        except Exception as e:
            raise ModelDownloadError(f"Failed to download models: {e}")
    
    def load_models(self) -> None:
        """Load all ONNX models into memory."""
        with self._lock:
            if self._loaded:
                return
            
            self.ensure_models_exist()
            
            model_dir = Path(self.config.model_dir)
            precision_suffix = self.config.get_precision_suffix()
            
            self.logger.info("Loading ONNX models...")
            
            # Text embedding (always FP32)
            self._load_session("text_embedding", model_dir / "text_embedding_fp32.onnx")
            
            # Audio processing models
            self._load_session("campplus", model_dir / "campplus.onnx")
            self._load_session("speech_tokenizer", model_dir / "speech_tokenizer_v3.onnx")
            
            # LLM models
            self._load_session("llm_backbone_initial", 
                               model_dir / f"llm_backbone_initial{precision_suffix}.onnx")
            self._load_session("llm_backbone_decode", 
                               model_dir / f"llm_backbone_decode{precision_suffix}.onnx")
            self._load_session("llm_decoder", 
                               model_dir / f"llm_decoder{precision_suffix}.onnx")
            self._load_session("llm_speech_embedding", 
                               model_dir / f"llm_speech_embedding{precision_suffix}.onnx")
            
            # Flow models
            self._load_session("flow_token_embedding", 
                               model_dir / f"flow_token_embedding{precision_suffix}.onnx")
            self._load_session("flow_speaker_projection", 
                               model_dir / f"flow_speaker_projection{precision_suffix}.onnx")
            self._load_session("flow_pre_lookahead", 
                               model_dir / f"flow_pre_lookahead{precision_suffix}.onnx")
            
            # Flow estimator has different naming
            estimator_name = f"flow.decoder.estimator.fp16.onnx" if "fp16" in precision_suffix else "flow.decoder.estimator.fp32.onnx"
            self._load_session("flow_estimator", model_dir / estimator_name)
            
            # HiFT models (always FP32 for numerical stability)
            self._load_session("hift_f0_predictor", model_dir / "hift_f0_predictor_fp32.onnx")
            self._load_session("hift_source_generator", model_dir / "hift_source_generator_fp32.onnx")
            self._load_session("hift_decoder", model_dir / "hift_decoder_fp32.onnx")
            
            # Load tokenizer
            self._load_tokenizer()
            
            self._loaded = True
            self.logger.info("All models loaded successfully!")
    
    def _load_session(self, name: str, path: Path) -> None:
        """Load a single ONNX session.
        
        Args:
            name: Session name for caching
            path: Path to ONNX file
        """
        if not path.exists():
            raise ModelNotFoundError(f"Model not found: {path}")
        
        self._sessions[name] = ort.InferenceSession(
            str(path),
            self._session_options,
            providers=self._providers
        )
        self.logger.debug(f"Loaded {name} from {path.name}")
    
    def _load_tokenizer(self) -> None:
        """Load the Qwen2 tokenizer."""
        from transformers import AutoTokenizer
        
        model_dir = Path(self.config.model_dir)
        
        # Try to load from local tokenizer files
        if (model_dir / "vocab.json").exists():
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), 
                trust_remote_code=True
            )
            self.logger.debug("Tokenizer loaded from local files")
        else:
            # Fall back to Qwen2 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-0.5B", 
                trust_remote_code=True
            )
            self.logger.debug("Tokenizer loaded from Qwen/Qwen2-0.5B")
    
    def get_session(self, name: str) -> ort.InferenceSession:
        """Get a loaded ONNX session by name.
        
        Args:
            name: Session name
            
        Returns:
            ONNX InferenceSession
            
        Raises:
            RuntimeError: If models not loaded
        """
        if not self._loaded:
            if self.config.lazy_load:
                self.load_models()
            else:
                raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if name not in self._sessions:
            raise ValueError(f"Unknown session: {name}")
        
        return self._sessions[name]
    
    @property
    def tokenizer(self):
        """Get the tokenizer."""
        if not self._loaded:
            if self.config.lazy_load:
                self.load_models()
            else:
                raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._tokenizer
    
    def unload_models(self) -> None:
        """Unload all models to free memory."""
        with self._lock:
            self._sessions.clear()
            self._tokenizer = None
            self._loaded = False
            self.logger.info("Models unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded
