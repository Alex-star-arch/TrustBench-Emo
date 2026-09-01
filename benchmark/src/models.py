"""Model wrappers for Qwen2-VL / Qwen2.5-VL and Qwen2.5 text models."""
import re
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .prompts import EMOTIONS, text_prompt, image_prompt, parse_output


class EmotionModel:
    """Generic wrapper around a HuggingFace model for TrustBench-Emo."""

    def __init__(self, model_path, kind, device=None):
        """
        Args:
            model_path: HuggingFace hub id or local path.
            kind: one of {'qwen2vl', 'qwen2_5_vl', 'qwen_text'}.
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
        elif self.kind == "qwen_text":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.eval()
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    def _confidence_from_logits(self, logits, input_ids, label_tokens):
        """Compute softmax probabilities over the seven emotion label tokens."""
        # logits: (seq_len, vocab_size); take last position
        last_logits = logits[-1, :]
        label_ids = torch.tensor(label_tokens, device=last_logits.device)
        probs = torch.softmax(last_logits, dim=-1)
        label_probs = probs[label_ids]
        return label_probs / label_probs.sum()  # re-normalise over the 7 labels

    def _label_tokens(self):
        """Map each emotion to its token id(s) in the model vocabulary."""
        if self.kind in ("qwen2vl", "qwen2_5_vl"):
            tok = self.processor.tokenizer
        else:
            tok = self.tokenizer
        label_tokens = []
        for emo in EMOTIONS:
            ids = tok.encode(emo, add_special_tokens=False)
            # use the first token of the label string
            label_tokens.append(ids[0] if ids else tok.unk_token_id)
        return label_tokens

    def generate(self, image_path=None, text=None):
        """Run inference and return a dict with label, rationale, confidence, raw output."""
        if self.kind in ("qwen2vl", "qwen2_5_vl"):
            if image_path:
                from PIL import Image
                prompt = image_prompt()
                messages = [{"role": "user", "content": [{"type": "image", "image": image_path},
                                                          {"type": "text", "text": prompt}]}]
            else:
                prompt = text_prompt(text)
                messages = [{"role": "user", "content": prompt}]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(images=Image.open(image_path) if image_path else None,
                                   text=text_input,
                                   return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False,
                                          max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
            generated_ids = out.sequences[0][len(inputs["input_ids"][0]):]
            raw = self.processor.batch_decode(generated_ids.unsqueeze(0), skip_special_tokens=True)[0]
            # confidence from the first generated token's score distribution
            label_tokens = self._label_tokens()
            scores = out.scores[0][0]
            confs = torch.softmax(scores, dim=-1)
            label_probs = confs[torch.tensor(label_tokens, device=scores.device)]
            label_probs = label_probs / label_probs.sum()
        else:
            prompt = text_prompt(text)
            messages = [{"role": "user", "content": prompt}]
            text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False,
                                          max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
            generated_ids = out.sequences[0][len(inputs["input_ids"][0]):]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            label_tokens = self._label_tokens()
            scores = out.scores[0][0]
            confs = torch.softmax(scores, dim=-1)
            label_probs = confs[torch.tensor(label_tokens, device=scores.device)]
            label_probs = label_probs / label_probs.sum()

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
