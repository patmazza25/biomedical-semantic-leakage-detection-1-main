# utils/cot_generator.py
from __future__ import annotations

import os
import re
import json
import torch
import requests
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Config & readiness
# -----------------------------------------------------------------------------

def _cfg(name: str):
    """Import a single name from config, falling back to env var silently."""
    try:
        import config as _cfg_mod
        return getattr(_cfg_mod, name, None) or None
    except Exception:
        return None

ANTHROPIC_API_KEY  = _cfg("ANTHROPIC_API_KEY")  or os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = _cfg("OPENAI_API_KEY")      or os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY     = _cfg("GOOGLE_API_KEY")      or os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = _cfg("OPENROUTER_API_KEY")  or os.getenv("OPENROUTER_API_KEY")
HF_TOKEN           = _cfg("HF_TOKEN")            or os.getenv("HF_TOKEN")

# Global for local model to avoid re-loading
_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_LEADING_MARK_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-•]\s*)")

def _postprocess_steps(text: str) -> List[str]:
    # Gemma 4 thinking mode cleanup: remove everything inside <|thought|> tags if present
    text = re.sub(r'<\|thought\|>.*?<\|channel\|>', '', text, flags=re.DOTALL)
    
    raw = (text or "").strip()
    if not raw:
        return []
    steps: List[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln: continue
        ln = _LEADING_MARK_RE.sub("", ln).strip()
        if ln: steps.append(ln)
    
    if len(steps) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", raw) if p.strip()]
        if len(parts) > 1: steps = parts
    return steps

def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def _mk_result(
    *,
    steps: List[str],
    provider: str,
    model: str,
    final: str = "",
    meta: Optional[Dict[str, Any]] = None,
    raw: Optional[Any] = None,
) -> Dict[str, Any]:
    return {
        "steps": steps,
        "final": final or "",
        "provider": provider,
        "model": model,
        "meta": _json_safe(meta or {}),
        "raw": _json_safe(raw),
    }

# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------

def _call_local_hf(question: str, model_id: str, temperature: float = 0.7) -> Optional[Dict[str, Any]]:
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is None:
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            print(f"Loading local model {model_id} onto GPU...")
            _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
            _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                token=HF_TOKEN,
                # use_flash_attention_2=True # Enable if flash-attn is installed
            )
        except Exception as e:
            print(f"Local Model Load Error: {e}")
            return None

    try:
        messages = [{"role": "user", "content": question}]
        # Gemma 4 Chat Template
        text = _LOCAL_PROCESSOR.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            # enable_thinking=True # Optional: enable for Gemma 4 native reasoning
        )
        inputs = _LOCAL_PROCESSOR(text=text, return_tensors="pt").to(_LOCAL_MODEL.device)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = _LOCAL_MODEL.generate(
                **inputs, 
                max_new_tokens=1024, 
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )
        
        response = _LOCAL_PROCESSOR.decode(outputs[0][input_len:], skip_special_tokens=True)
        steps = _postprocess_steps(response)
        return _mk_result(steps=steps, provider="local_hf", model=model_id, raw=response)
    except Exception as e:
        print(f"Local Generation Error: {e}")
        return None

def _call_openrouter(question: str, model: str = "anthropic/claude-haiku-4-5", temperature: float = 0.7) -> Optional[Dict[str, Any]]:
    if not OPENROUTER_API_KEY: return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": question}], "temperature": temperature}
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        res = resp.json()
        text = res["choices"][0]["message"]["content"]
        return _mk_result(steps=_postprocess_steps(text), provider="openrouter", model=model, raw=res)
    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return None

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate(question: str, prefer: str = "local", model: Optional[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
    """Generate CoT steps for a question."""
    
    # Priority 1: Local HF (if specified or prefer is local)
    if prefer == "local" or (model and not "/" in model):
        model_id = model or "google/gemma-4-31B-it"
        res = _call_local_hf(question, model_id=model_id, temperature=temperature)
        if res: return res

    # Priority 2: OpenRouter
    if prefer == "openrouter" or (model and "/" in model):
        model_id = model or "openai/gpt-4o-mini"
        res = _call_openrouter(question, model=model_id, temperature=temperature)
        if res: return res

    return _mk_result(
        steps=["ERROR: All generation paths failed.", "Check GPU memory, HF_TOKEN, or API keys."],
        provider="error", model="none"
    )
