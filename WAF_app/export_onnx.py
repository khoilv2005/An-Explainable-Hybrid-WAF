import os
import json
import torch

from models.model import WAF_Attention_Model

MAX_LEN = 500
EMBEDDING_DIM = 128

def _infer_vocab_size_from_state(state: dict) -> int:
    # Try common embedding parameter names
    for key in ("embedding.weight", "emb.weight", "token_embedding.weight"):
        if key in state and state[key].dim() == 2:
            return state[key].size(0)
    # Fallback: search any 2D tensor with second dim == EMBEDDING_DIM
    for tensor in state.values():
        try:
            if hasattr(tensor, "dim") and tensor.dim() == 2 and tensor.size(1) == EMBEDDING_DIM:
                return tensor.size(0)
        except Exception:
            pass
    raise ValueError("Unable to infer vocab_size from state_dict; provide tokenizer JSON.")

def export_to_onnx(model_path: str, onnx_path: str, tokenizer_json_path: str | None = None) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Determine vocab_size: prefer JSON word_index, else infer from state_dict
    vocab_size = None
    if tokenizer_json_path and os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect either full tokenizer config or just word_index mapping
        word_index = data.get("word_index", data)
        if not isinstance(word_index, dict):
            raise ValueError("Invalid tokenizer JSON: expected a dict or {word_index: {...}}")
        vocab_size = int(len(word_index)) + 1
        print(f"Loaded word_index from JSON. vocab_size={vocab_size}")

    state = torch.load(model_path, map_location=torch.device("cpu"))
    if vocab_size is None:
        vocab_size = _infer_vocab_size_from_state(state)
        print(f"Inferred vocab_size from state_dict: {vocab_size}")

    model = WAF_Attention_Model(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, num_classes=1)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (1, MAX_LEN), dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {onnx_path}")


if __name__ == "__main__":
    model_path = os.getenv("ML_MODEL_PATH", "/app/models/waf_model.pth")
    onnx_path = os.getenv("ML_MODEL_ONNX_PATH", "/app/models/waf_model.onnx")
    tokenizer_json_path = os.getenv("ML_TOKENIZER_JSON_PATH")
    export_to_onnx(model_path, onnx_path, tokenizer_json_path)
