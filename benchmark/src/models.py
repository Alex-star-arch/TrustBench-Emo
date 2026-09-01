"""Model wrappers for TrustBench-Emo.

Supported ``kind`` values:
  - ``qwen2vl``, ``qwen2_5_vl`` : Qwen vision-language models
  - ``qwen_text``, ``llama``, ``gemma``, ``causal`` : decoder-only text / causal LMs
    (loaded via ``AutoModelForCausalLM`` + ``AutoTokenizer``)
  - ``llava`` : LLaVA-1.5 vision-language model

All wrappers expose a single ``generate(image_path=..., text=...)`` method that
returns ``{pred_label, rationale, confidence, raw}``. Confidence is the
renormalised softmax probability of the seven emotion-label tokens at the first
generated position, consistent across every model family so D1 (calibration) is
comparable.
"""
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

from .prompts import EMOTIONS, text_prompt, image_prompt, parse_output

_VL_KINDS = ("qwen2vl", "qwen2_5_vl", "llava")
_CAUSAL_KINDS = ("qwen_text", "llama", "gemma", "causal")


class EmotionModel:
    """Generic wrapper around a HuggingFace model for TrustBench-Emo."""

    def __init__(self, model_path, kind, device=None):
        """
        Args:
            model_path: HuggingFace hub id or local path.
            kind: one of the supported kinds (see module docstring).
            device: torch device; defaults to cuda:0 if available.
        """
        self.model_path = model_path
        self.kind = kind
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if self.kind in ("qwen2vl", "qwen2_5_vl"):
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
            self.model.eval()
        elif self.kind in _CAUSAL_KINDS:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.eval()
        elif self.kind == "llava":
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.processor = LlavaProcessor.from_pretrained(self.model_path)
            self.model.eval()
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    # ------------------------------------------------------------------ helpers
    def _tokenizer(self):
        return self.processor.tokenizer if self.kind in _VL_KINDS else self.tokenizer

    def _label_tokens(self):
        """Map each emotion to its first token id in the model vocabulary."""
        tok = self._tokenizer()
        label_tokens = []
        for emo in EMOTIONS:
            ids = tok.encode(emo, add_special_tokens=False)
            label_tokens.append(ids[0] if ids else tok.unk_token_id)
        return label_tokens

    def _finalize(self, out, input_len, label_tokens):
        """Slice generated ids, decode to text, and compute label-probabilities."""
        generated = out.sequences[0][input_len:]
        tok = self._tokenizer()
        raw = tok.decode(generated, skip_special_tokens=True)
        scores = out.scores[0][0]  # logits for the first generated token
        confs = torch.softmax(scores, dim=-1)
        lp = confs[torch.tensor(label_tokens, device=scores.device)]
        label_probs = lp / lp.sum()
        return raw, label_probs

    def _postprocess(self, raw, label_probs):
        pred_label, rationale = parse_output(raw)
        if pred_label is None:
            conf = float(label_probs.max())
            pred_label = EMOTIONS[int(label_probs.argmax())]
        else:
            conf = float(label_probs[EMOTIONS.index(pred_label)])
        return {
            "pred_label": pred_label,
            "rationale": rationale,
            "confidence": conf,
            "raw": raw,
        }

    # ------------------------------------------------------------------ dispatch
    def generate(self, image_path=None, text=None):
        """Run inference; returns {pred_label, rationale, confidence, raw}."""
        if self.kind in ("qwen2vl", "qwen2_5_vl"):
            return self._generate_qwen_vl(image_path, text)
        if self.kind == "llava":
            return self._generate_llava(image_path, text)
        return self._generate_causal(text)

    def _generate_qwen_vl(self, image_path, text):
        from PIL import Image

        if image_path:
            prompt = image_prompt()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}]}]
        else:
            prompt = text_prompt(text)
            messages = [{"role": "user", "content": prompt}]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            images=Image.open(image_path) if image_path else None,
            text=text_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=128,
                return_dict_in_generate=True, output_scores=True)
        raw, label_probs = self._finalize(out, len(inputs["input_ids"][0]), self._label_tokens())
        return self._postprocess(raw, label_probs)

    def _generate_llava(self, image_path, text):
        from PIL import Image

        if image_path:
            prompt = f"<image>\n{image_prompt()}"
        else:
            prompt = text_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path) if image_path else None
        inputs = self.processor(
            images=image, text=text_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=128,
                return_dict_in_generate=True, output_scores=True)
        raw, label_probs = self._finalize(out, len(inputs["input_ids"][0]), self._label_tokens())
        return self._postprocess(raw, label_probs)

    def _generate_causal(self, text):
        prompt = text_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=128,
                return_dict_in_generate=True, output_scores=True)
        raw, label_probs = self._finalize(out, len(inputs["input_ids"][0]), self._label_tokens())
        return self._postprocess(raw, label_probs)
