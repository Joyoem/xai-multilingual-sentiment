from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, List, Mapping, Sequence


EmotionVector = List[List[float]]
RawModelOutput = Sequence[float] | Mapping[str, object]


class SentimentWrapper:
    """Unified wrapper that always returns a (batch, 6) emotion probability matrix.

    Label order follows BRIGHTER Track-A emotion dimensions (each dimension
    is binary in the original annotation scheme):
    (anger, disgust, fear, joy, sadness, surprise).
    """

    labels = ("anger", "disgust", "fear", "joy", "sadness", "surprise")

    def __init__(
        self,
        backend_name: str,
        predictor: Callable[[str], RawModelOutput] | None = None,
    ):
        self.backend_name = backend_name
        self.predictor = predictor or self._heuristic_predictor

    def predict(self, text: str | Sequence[str]) -> EmotionVector:
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, Sequence):
            texts = list(text)
            if any(not isinstance(item, str) for item in texts):
                raise TypeError("Sequence inputs to SentimentWrapper.predict must contain only strings.")
        else:
            raise TypeError("SentimentWrapper.predict expects a string or a sequence of strings.")
        if not texts:
            return []

        probs_batch: EmotionVector = []
        for item in texts:
            raw = self.predictor(item)
            probs_batch.append(self._to_probabilities(raw))
        return probs_batch

    def _to_probabilities(self, raw: RawModelOutput) -> List[float]:
        if self._is_llama_style(raw):
            return _normalize_to_probability(self._llama_yesno_to_distribution(raw))
        if isinstance(raw, Mapping):
            raise ValueError("Mapping outputs are only supported for llama-style yes/no predictors.")

        values = [float(x) for x in list(raw)]
        if len(values) != len(self.labels):
            raise ValueError(f"Expected {len(self.labels)} outputs, got {len(values)} from {self.backend_name}")

        if any(v < 0.0 or v > 1.0 for v in values):
            values = [_sigmoid(v) for v in values]
        return _normalize_to_probability(values)

    def _is_llama_style(self, raw: RawModelOutput) -> bool:
        if not isinstance(raw, Mapping):
            return False
        if "llama" in self.backend_name.lower():
            return True

        lowered = {str(k).lower(): v for k, v in raw.items()}
        for label in self.labels:
            value = lowered.get(label)
            if isinstance(value, Mapping):
                value_keys = {str(k).lower() for k in value.keys()}
                if "yes" in value_keys or "no" in value_keys or "1" in value_keys or "0" in value_keys:
                    return True
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
                return True
        return False

    def _llama_yesno_to_distribution(self, raw: RawModelOutput) -> List[float]:
        if not isinstance(raw, Mapping):
            raise ValueError("Llama mapping expects dict-style per-label yes/no outputs.")

        output: List[float] = []
        lowered_map = {str(k).lower(): v for k, v in raw.items()}
        for label in self.labels:
            value = lowered_map.get(label, 0.0)
            output.append(_extract_yes_probability(value))
        return output

    def _heuristic_predictor(self, text: str) -> Sequence[float]:
        lower = text.lower()
        logits = [0.0] * len(self.labels)
        keyword_map = {
            0: ("angry", "mad", "furious", "hate"),
            1: ("disgust", "gross", "nasty", "revolting"),
            2: ("fear", "scared", "afraid", "terrified"),
            3: ("love", "great", "happy", "excellent"),
            4: ("sad", "cry", "down", "depressed"),
            5: ("wow", "surprised", "unexpected", "amazing"),
        }
        for idx, words in keyword_map.items():
            logits[idx] += sum(1.0 for word in words if word in lower)
        return logits


def build_mdeberta_predictor(model_name: str = "microsoft/mdeberta-v3-base") -> Callable[[str], Sequence[float]]:
    """Build a predictor function that uses mDeBERTa model for 6-class emotion prediction.

    Args:
        model_name: HuggingFace model ID (default: microsoft/mdeberta-v3-base)
                   For Kaggle/limited compute, consider:
                   - "microsoft/mdeberta-v3-small" (~100M params, fastest)
                   - "xlm-roberta-base" (~270M params, good balance)

    Returns:
        A predictor function that takes text and returns 6 emotion logits.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for mDeBERTa predictor. "
            "Install with: pip install transformers torch."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict_fn(text: str) -> Sequence[float]:
        """Predict emotion logits for text."""
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors="pt"
        )
        # Move tensors to same device as model
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits[0]  # Shape: (6,)
            return logits.cpu().numpy().tolist()

    return predict_fn


def build_llama_predictor(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    use_vllm: bool = False,
    device_map: str = "auto"
) -> Callable[[str], Mapping[str, object]]:
    """Build a predictor function that uses Llama for emotion Yes/No classification.

    Args:
        model_name: HuggingFace model ID or local path.
                   
                   RECOMMENDED FOR KAGGLE (8B or smaller):
                   - "meta-llama/Llama-2-7b-hf" (7B, fits in 16GB GPU) ⭐
                   - "microsoft/phi-2" (2.7B, very efficient)
                   - "mistralai/Mistral-7B-v0.1" (7B, good accuracy)
                   
                   NOT RECOMMENDED FOR KAGGLE:
                   - "meta-llama/Llama-3-8b" (needs 20GB+ GPU memory)
                   - Any 70B+ models (requires multi-GPU)
        
        use_vllm: If True, use vLLM for faster inference.
                 For Kaggle, usually False is more reliable (vLLM requires extra setup).
        
        device_map: "auto" for automatic device placement, or specific device string
                   "auto" works best on Kaggle.

    Returns:
        A predictor function that returns dict with emotion -> {"yes": p, "no": 1-p}
    """
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is required for optimized Llama inference. "
                "Install with: pip install vllm."
            ) from exc

        llm = LLM(model=model_name, dtype="auto", max_model_len=512)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

        def predict_fn_vllm(text: str) -> Mapping[str, object]:
            emotions = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
            results = {}

            for emotion in emotions:
                prompt = f'Does this text contain {emotion}? Text: "{text}"\nAnswer (Yes/No):'
                output = llm.generate([prompt], sampling_params)[0]
                response = output.outputs[0].text.strip().lower()
                yes_prob = 1.0 if "yes" in response else (0.5 if "maybe" in response or "both" in response else 0.0)
                results[emotion] = {"yes": yes_prob, "no": 1.0 - yes_prob}

            return results

        return predict_fn_vllm

    else:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for Llama predictor. "
                "Install with: pip install transformers torch."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token if not defined (common for causal LMs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try 8-bit quantization to reduce memory footprint (optional)
        load_in_8bit = False
        try:
            import bitsandbytes  # noqa: F401
            load_in_8bit = True
        except ImportError:
            pass
        
        try:
            if load_in_8bit:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            else:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
        except (RuntimeError, OSError) as e:
            warnings.warn(
                f"Failed to load {model_name} with device_map='{device_map}': {e}. "
                f"Falling back to CPU-only mode.",
                RuntimeWarning,
                stacklevel=2
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def predict_fn_transformers(text: str) -> Mapping[str, object]:
            emotions = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
            results = {}

            for emotion in emotions:
                prompt = f'Does this text contain {emotion}? Text: "{text}"\nAnswer (Yes/No):'
                
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(device)

                    with torch.no_grad():
                        # Use generate for better control over output
                        output_ids = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=5,
                            temperature=0.0,
                            top_p=1.0,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
                    yes_prob = 1.0 if "yes" in response else (0.5 if "maybe" in response or "both" in response else 0.0)
                    results[emotion] = {"yes": yes_prob, "no": 1.0 - yes_prob}
                    
                except (RuntimeError, ValueError) as e:
                    warnings.warn(
                        f"Error predicting emotion '{emotion}': {e}. Using default 0.5 probability.",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    results[emotion] = {"yes": 0.5, "no": 0.5}

            return results

        return predict_fn_transformers


def _extract_yes_probability(value: object) -> float:
    if isinstance(value, Mapping):
        lowered = {str(k).lower(): v for k, v in value.items()}
        yes = float(lowered.get("yes", lowered.get("1", 0.0)))
        inferred_no = 1.0 - yes
        no = float(lowered.get("no", lowered.get("0", inferred_no)))
        denom = yes + no
        return yes / denom if denom > 0 else 0.0

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        arr = list(value)
        if len(arr) == 2:
            yes = float(arr[0])
            no = float(arr[1])
            denom = yes + no
            return yes / denom if denom > 0 else 0.0
        if len(arr) == 1:
            return float(arr[0])

    return float(value)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize_to_probability(raw: Iterable[float]) -> List[float]:
    values = list(float(x) for x in raw)
    if not values:
        raise ValueError("Cannot normalize an empty vector")

    min_v = min(values)
    if min_v < 0:
        values = [v - min_v for v in values]
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)

    probs = [v / total for v in values]
    probs_sum = sum(probs)
    if not math.isclose(probs_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        warnings.warn(
            "Probability vector sum deviated from 1.0 due to floating-point precision; corrected final element.",
            RuntimeWarning,
            stacklevel=2,
        )
        probs[-1] = max(0.0, 1.0 - sum(probs[:-1]))
        corrected_sum = sum(probs)
        if any(p < 0.0 or p > 1.0 for p in probs) or not math.isclose(corrected_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Failed to produce a valid probability distribution after correction.")
    return probs
