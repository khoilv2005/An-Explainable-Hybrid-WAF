"""
ML Predictor Module for WAF
Using PyTorch WAF_Attention_Model with character-level tokenization
Optimized for high-performance inference
"""
import os
import pickle
import json
import logging
from typing import Tuple
from functools import lru_cache

import numpy as np

try:
    import onnxruntime as ort
    ONNX_RT_AVAILABLE = True
except ImportError:
    ONNX_RT_AVAILABLE = False

# Lazy import torch - only when needed
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

logger = logging.getLogger(__name__)

# Use the trained architecture from models/model.py to match saved checkpoints
# Lazy import - only when PyTorch fallback is needed
_WAF_Model = None
def get_waf_model_class():
    global _WAF_Model
    if _WAF_Model is None:
        from models.model import WAF_Attention_Model
        _WAF_Model = WAF_Attention_Model
    return _WAF_Model

# =============================================================================
# CONFIGURATION - MUST MATCH TRAINING
# =============================================================================
MAX_LEN = 500
EMBEDDING_DIM = 128

# Lazy device detection
_DEVICE = None
def get_device():
    global _DEVICE
    if _DEVICE is None:
        torch = get_torch()
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

ONNX_MODEL_PATH = os.getenv('ML_MODEL_ONNX_PATH', '/app/models/waf_model.onnx')
FORCE_ONNX = os.getenv('ML_FORCE_ONNX', 'false').lower() == 'true'

# =============================================================================
# XAI - DANGEROUS PATTERNS FOR EXPLAINABILITY
# =============================================================================
DANGEROUS_PATTERNS = {
    'SQL': [
        'select', 'union', 'drop', 'insert', 'delete', 'update', 'exec', 'execute',
        'waitfor', 'sleep', 'benchmark', 'information_schema', 'xp_cmdshell',
        "' or ", '" or ', 'or 1=1', 'or 0=0', "'--", '"--', '1=1', '1 = 1',
        "or '1'='1", 'or "1"="1"', ' or 1', "' or '", '" or "', '= 1 --'
    ],
    'XSS': [
        '<script', '</script', 'javascript:', 'onerror=', 'onload=', 'onclick=',
        'alert(', 'eval(', 'document.cookie', '<iframe', '<svg', '<img',
        'onfocus=', 'onmouseover=', 'prompt(', 'confirm('
    ],
    'CMD': [
        'etc/passwd', 'etc/shadow', 'win.ini', 'whoami', 'cat ', 'ls ',
        'rm -rf', 'ping ', 'curl ', 'wget ', '$(', '`', '&&', '||', 
        '/bin/sh', '/bin/bash', 'cmd.exe', 'powershell'
    ],
    'PATH': [
        '../', '..\\', '%2e%2e%2f', '%2e%2e\\', '..../', '....\\',
        '/etc/', 'C:\\', 'Windows\\'
    ]
}

@lru_cache(maxsize=2048)
def detect_patterns_cached(text: str) -> tuple:
    """Detect dangerous patterns in text for XAI - cached"""
    text_lower = text.lower()
    detected = {}

    for category, patterns in DANGEROUS_PATTERNS.items():
        found = tuple(p for p in patterns if p.lower() in text_lower)
        if found:
            detected[category] = found

    # Convert to tuple for caching
    return tuple((k, v) for k, v in detected.items())

def detect_patterns(text: str) -> dict:
    """Detect dangerous patterns in text for XAI"""
    # Truncate for cache efficiency
    cache_key = text[:500] if len(text) > 500 else text
    cached = detect_patterns_cached(cache_key)
    return dict(cached)


# =============================================================================
# ML PREDICTOR CLASS
# =============================================================================
class MLPredictor:
    """Machine Learning predictor for WAF using PyTorch model - optimized"""

    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Initialize ML Predictor

        Args:
            model_path: Path to PyTorch model (.pth file)
            tokenizer_path: Path to tokenizer (.pkl file)
        """
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_loaded = False
        self.vocab_size = 0
        self.use_onnx = False
        self.onnx_session = None
        self.onnx_path = ONNX_MODEL_PATH
        # Pre-allocated numpy array for ONNX inference
        self._onnx_input_buffer = None

        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from disk"""
        try:
            # 1. Load tokenizer - prioritize JSON (faster, no keras dependency)
            tokenizer_json_path = os.getenv('ML_TOKENIZER_JSON_PATH')

            # Try JSON first if available
            if tokenizer_json_path and os.path.exists(tokenizer_json_path):
                try:
                    with open(tokenizer_json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                    word_index = data.get('word_index', data)
                    if not isinstance(word_index, dict):
                        raise ValueError('Invalid tokenizer JSON')

                    class SimpleCharTokenizer:
                        def __init__(self, word_index: dict):
                            self.word_index = word_index
                        def texts_to_sequences(self, texts):
                            return [[self.word_index.get(ch, 0) for ch in t] for t in texts]

                    self.tokenizer = SimpleCharTokenizer(word_index)
                    self.vocab_size = len(word_index) + 1
                    logger.info(f"✓ Loaded tokenizer from JSON (vocab size: {self.vocab_size})")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer JSON: {e}")
                    self.tokenizer = None

            # Fallback to pickle if JSON not available
            if self.tokenizer is None and os.path.exists(self.tokenizer_path):
                try:
                    with open(self.tokenizer_path, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                    self.vocab_size = len(self.tokenizer.word_index) + 1
                    logger.info(f"✓ Loaded tokenizer from pickle (vocab size: {self.vocab_size})")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer pickle: {e}")
                    self.tokenizer = None

            if self.tokenizer is None:
                logger.error("No tokenizer available")
                return False

            # 2. Load ONNX if available (preferred for performance)
            if ONNX_RT_AVAILABLE and os.path.exists(self.onnx_path):
                try:
                    # Optimized session options for single-threaded inference
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    self.onnx_session = ort.InferenceSession(
                        self.onnx_path,
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"]
                    )
                    self.use_onnx = True
                    # Pre-allocate input buffer
                    self._onnx_input_buffer = np.zeros((1, MAX_LEN), dtype=np.int64)
                    logger.info(f"✓ Loaded ONNX model from {self.onnx_path} (optimized)")
                    self.is_loaded = True
                    return True
                except Exception as e:
                    if FORCE_ONNX:
                        logger.error(f"Failed to load ONNX model and ML_FORCE_ONNX=true: {e}")
                        self.is_loaded = False
                        return False
                    logger.error(f"Failed to load ONNX model, falling back to PyTorch: {e}")
            elif FORCE_ONNX:
                logger.error(f"ML_FORCE_ONNX=true but ONNX model not found at {self.onnx_path}")
                self.is_loaded = False
                return False

            # 3. Load PyTorch model (fallback)
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False

            torch = get_torch()
            DEVICE = get_device()
            WAF_Attention_Model = get_waf_model_class()

            self.model = WAF_Attention_Model(
                vocab_size=self.vocab_size,
                embedding_dim=EMBEDDING_DIM,
                num_classes=1
            )
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            logger.info(f"✓ Loaded PyTorch model from {self.model_path}")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def _pad_sequences(self, sequences, maxlen, padding='post', truncating='post'):
        """Pad sequences to same length (reimplemented from Keras)"""
        result = []
        for seq in sequences:
            if len(seq) > maxlen:
                if truncating == 'post':
                    seq = seq[:maxlen]
                else:
                    seq = seq[-maxlen:]
            elif len(seq) < maxlen:
                if padding == 'post':
                    seq = seq + [0] * (maxlen - len(seq))
                else:
                    seq = [0] * (maxlen - len(seq)) + seq
            result.append(seq)
        return result
    
    def predict(self, request_data: str) -> Tuple[str, float]:
        """
        Predict if request is normal or attack - optimized for speed

        Args:
            request_data: Raw request string

        Returns:
            Tuple of (prediction, confidence) or (prediction, confidence, patterns)
        """
        if not self.is_loaded:
            return ('unknown', 0.0)

        try:
            sequences = self.tokenizer.texts_to_sequences([request_data])
            padded = self._pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

            if self.use_onnx and self.onnx_session:
                # Use pre-allocated buffer for ONNX - avoid allocation
                if self._onnx_input_buffer is not None:
                    self._onnx_input_buffer[0, :] = padded[0]
                    inputs = self._onnx_input_buffer
                else:
                    inputs = np.asarray(padded, dtype=np.int64)
                logits = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inputs})[0]
                attack_prob = float(1.0 / (1.0 + np.exp(-logits[0][0])))
            else:
                torch = get_torch()
                DEVICE = get_device()
                tensor = torch.LongTensor(padded).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(tensor)
                    attack_prob = torch.sigmoid(logits).cpu().numpy()[0][0]

            prediction_label = 'attack' if attack_prob >= 0.5 else 'normal'
            confidence = float(attack_prob)

            # XAI - Detect dangerous patterns (cached)
            detected_patterns = detect_patterns(request_data)

            return (prediction_label, confidence, detected_patterns) if detected_patterns else (prediction_label, confidence)

        except Exception as e:
            logger.error(f"Error during ML prediction: {e}")
            return ('error', 0.0)
    
    def get_request_features(self, req) -> str:
        """
        Extract features from Flask request object
        """
        try:
            features = []
            
            if req.path:
                features.append(req.path)
            
            if req.query_string:
                features.append(req.query_string.decode('utf-8', errors='ignore'))
            
            if req.data:
                features.append(req.data.decode('utf-8', errors='ignore'))
            
            if req.form:
                for key, value in req.form.items():
                    features.append(f"{key}={value}")
            
            return ' '.join(features)
            
        except Exception as e:
            logger.error(f"Error extracting request features: {e}")
            return ""
    
    def _predict_proba_for_lime(self, texts):
        """
        Wrapper for LIME explainer - returns raw logits for better weight differentiation
        
        Note: Using raw logits instead of sigmoid probabilities because
        sigmoid saturates at extreme values (0.99+), making perturbation
        effects too small for meaningful LIME weights.
        
        IMPORTANT: Process texts in batch (not loop) to match explain.py behavior
        """
        # Batch tokenization (like explain.py)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = self._pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        
        if self.use_onnx and self.onnx_session:
            inputs = np.asarray(padded, dtype=np.int64)
            logits = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inputs})[0]
            # logits shape: (batch_size, 1)
        else:
            torch = get_torch()
            DEVICE = get_device()
            tensor = torch.LongTensor(padded).to(DEVICE)
            with torch.no_grad():
                logits = self.model(tensor).cpu().numpy()
        
        # Convert to [p_normal, p_attack] format (like explain.py)
        results = []
        for p in logits:
            p_attack = float(p[0])  # Raw logit
            p_normal = 1 - p_attack
            results.append([p_normal, p_attack])

        return np.array(results)
    
    def explain(self, request_data: str, num_samples: int = 500) -> dict:
        """
        Generate LIME explanation for prediction
        
        Args:
            request_data: Raw request string
            num_samples: Number of samples for LIME (more = accurate but slower)
        
        Returns:
            Dictionary with explanation data
        """
        if not self.is_loaded:
            return None
        
        try:
            from lime.lime_text import LimeTextExplainer
            
            # Create character-level explainer (matches model tokenization)
            explainer = LimeTextExplainer(
                class_names=["Normal", "Attack"],
                char_level=True,
                split_expression=lambda x: list(x),
                bow=False
            )
            
            # Get explanation - explain ALL unique characters (no limit)
            exp = explainer.explain_instance(
                request_data,
                self._predict_proba_for_lime,
                num_features=len(set(request_data)),  # All unique chars
                num_samples=num_samples
            )
            
            # Get actual probabilities for display (using sigmoid)
            # Note: _predict_proba_for_lime returns raw logits for better LIME weights
            # but we need sigmoid probabilities for display
            sequences = self.tokenizer.texts_to_sequences([request_data])
            padded = self._pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
            
            if self.use_onnx and self.onnx_session:
                inputs = np.asarray(padded, dtype=np.int64)
                logits = self.onnx_session.run(None, {self.onnx_session.get_inputs()[0].name: inputs})[0]
                p_attack = float(1 / (1 + np.exp(-logits[0][0])))  # Sigmoid for display
            else:
                torch = get_torch()
                DEVICE = get_device()
                tensor = torch.LongTensor(padded).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(tensor)
                    p_attack = float(torch.sigmoid(logits).cpu().numpy()[0][0])  # Sigmoid for display
            
            p_normal = 1 - p_attack
            
            # Get character weights
            char_weights = {}
            for char, weight in exp.as_list():
                if char in char_weights:
                    char_weights[char] += weight
                else:
                    char_weights[char] = weight
            
            # Sort by importance (absolute weight)
            sorted_chars = sorted(char_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Get top dangerous (positive weight) and safe (negative weight) - increased to 10
            top_dangerous = [(c, w) for c, w in sorted_chars if w > 0][:10]
            top_safe = [(c, w) for c, w in sorted_chars if w < 0][:10]
            
            # N-gram analysis (3-grams) for better interpretability
            ngram_weights = []
            n = 3
            for i in range(len(request_data) - n + 1):
                ngram = request_data[i:i+n]
                weight = sum(char_weights.get(c, 0) for c in ngram)
                ngram_weights.append((ngram, weight))
            
            # Sort n-grams by absolute weight
            sorted_ngrams = sorted(ngram_weights, key=lambda x: abs(x[1]), reverse=True)
            top_dangerous_ngrams = [(ng, w) for ng, w in sorted_ngrams if w > 0][:5]
            top_safe_ngrams = [(ng, w) for ng, w in sorted_ngrams if w < 0][:5]
            
            return {
                'payload': request_data[:100],
                'p_normal': p_normal,
                'p_attack': p_attack,
                'prediction': 'Attack' if p_attack > 0.5 else 'Normal',
                'confidence': max(p_normal, p_attack),
                'top_dangerous': top_dangerous,
                'top_safe': top_safe,
                'top_dangerous_ngrams': top_dangerous_ngrams,
                'top_safe_ngrams': top_safe_ngrams,
                'all_weights': char_weights
            }
            
        except Exception as e:
            logger.error(f"Error during LIME explanation: {e}")
            return None
    
    def log_explanation(self, request_data: str, num_samples: int = 200):
        """Generate and log LIME explanation"""
        exp = self.explain(request_data, num_samples)
        if not exp:
            return
        
        # Build log message
        lines = []
        lines.append("=" * 70)
        lines.append("🔍 PHÂN TÍCH PAYLOAD (LIME XAI)")
        lines.append("=" * 70)
        lines.append(f"\n📝 Payload: {exp['payload']}...")
        lines.append(f"\n📊 Dự đoán của Model:")
        lines.append(f"   🟢 Bình thường: {exp['p_normal']*100:.2f}%")
        lines.append(f"   🔴 Tấn công:    {exp['p_attack']*100:.2f}%")
        
        verdict = "⚠️ NGUY HIỂM" if exp['p_attack'] > 0.5 else "✅ AN TOÀN"
        lines.append(f"   {verdict} (Confidence: {exp['confidence']*100:.2f}%)")
        
        if exp['top_dangerous']:
            lines.append(f"\n💀 Top Ký tự Nguy hiểm (🔴):")
            for char, weight in exp['top_dangerous']:
                char_disp = repr(char) if char in ' \t\n' else f"'{char}'"
                lines.append(f"   {char_disp:6s} | Weight: {weight:+.4f}")
        
        if exp['top_safe']:
            lines.append(f"\n🛡️ Top Ký tự An toàn (🟢):")
            for char, weight in exp['top_safe']:
                char_disp = repr(char) if char in ' \t\n' else f"'{char}'"
                lines.append(f"   {char_disp:6s} | Weight: {weight:+.4f}")
        
        lines.append("=" * 70)
        
        # Log each line
        for line in lines:
            logger.info(f"[XAI] {line}")
        
        return exp
    
    def reload_model(self):
        """Reload model from disk"""
        logger.info("Reloading ML model...")
        return self._load_model()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================
_ml_predictor = None

def get_ml_predictor() -> MLPredictor:
    """Get or create global ML predictor instance"""
    global _ml_predictor
    
    if _ml_predictor is None:
        model_path = os.getenv('ML_MODEL_PATH', '/app/models/waf_model.pth')
        tokenizer_path = os.getenv('ML_TOKENIZER_PATH', '/app/models/tokenizer.pkl')
        
        _ml_predictor = MLPredictor(model_path, tokenizer_path)
    
    return _ml_predictor

