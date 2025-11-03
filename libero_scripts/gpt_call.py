# instruction_variants.py
import json
import re
import os
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
# ---- model_infer.py (drop-in) ----
import time, random, traceback
from typing import Dict, List, Optional, Tuple


# ----------- Prepare dict_config_m -----------
# Replace with your own model setup if you already have dict_config_m defined elsewhere.
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
dict_config_m = {"client": client, "model": "gpt-5-2025-08-07"}

# If you already have these in another file, remove these fallbacks and import instead.
def postprocess_output(text: Optional[str]):
    """Basic cleanup: strip whitespace, remove surrounding code fences."""
    if text is None:
        return ""
    s = str(text).strip()
    # strip simple fenced blocks like ```json ... ```
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`")
        # Remove first line language tag if present (e.g., json, python)
        if "\n" in s:
            first_nl = s.find("\n")
            s = s[first_nl + 1 :].strip()
    return s

def ensure_prepare_prompt_exists():
    """Return a reference to prepare_prompt().
    If the user already defined it, use that; otherwise provide a compact fallback.
    """
    if "prepare_prompt" in globals():
        return globals()["prepare_prompt"]

    def _fallback_prepare_prompt(prompt: str, image=None, conversation_history=None, **kwargs):
        # Minimal compatible fallback with your message format
        if conversation_history is None:
            content = []
            if image is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                })
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = list(conversation_history)
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        return messages

    return _fallback_prepare_prompt


def _maybe_make_openai_client(dict_config_m: Dict):
    """
    If dict_config_m lacks 'client' and OpenAI SDK is available with an API key,
    instantiate a client and return (client, model). Otherwise return existing.
    """
    client = dict_config_m.get("client")
    model = dict_config_m.get("model")
    if client is not None and model:
        return client, model

    # Try to create an OpenAI client only if the package and key are available.
    try:
        from openai import OpenAI  # type: ignore
        import os
        api_key = dict_config_m.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = dict_config_m.get("base_url")  # optional, for proxies / custom gateways
        if not api_key:
            # No API key → leave as is; caller provided some other client or will error later
            return client, model
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        if not model:
            model = "gpt-4o-mini"  # sensible default
        dict_config_m["client"] = client
        dict_config_m["model"]  = model
        return client, model
    except Exception:
        # openai not installed or cannot init; return what we have
        return client, model


def model_infer(
    dict_input: Dict,
    dict_config_m: Dict,
    conversation_history: Optional[List[Dict]] = None,
    print_content: bool = False,
    **kwargs
) -> Tuple[str, List[Dict]]:
    """
    Generic inference wrapper compatible with your message format.

    dict_config_m:
      - REQUIRED: 'client' (with .chat.completions.create) OR OpenAI SDK installed + API key
      - REQUIRED: 'model' (e.g., 'gpt-4o-mini')
      - OPTIONAL gen params: 'temperature', 'top_p', 'max_tokens', 'seed', 'timeout'

    dict_input:
      - 'prompt': str               # required
      - 'image': tuple(image, image_data_base64) OR None  # optional

    Returns:
      (content: str, conversation: List[Dict]])
    """
    # Resolve client + model
    client, model = _maybe_make_openai_client(dict_config_m)
    if client is None or not model:
        raise RuntimeError(
            "No valid client/model. Provide dict_config_m['client'] and ['model'], "
            "or install `openai` and set OPENAI_API_KEY."
        )

    # Pull optional gen params
    temperature = dict_config_m.get("temperature", kwargs.get("temperature", 0.2))
    top_p       = dict_config_m.get("top_p", kwargs.get("top_p", 1.0))
    seed        = dict_config_m.get("seed", kwargs.get("seed", None))
    timeout     = dict_config_m.get("timeout", kwargs.get("timeout", None))

    # Prepare prompt/messages
    prompt = dict_input.get("prompt", "")
    img_out = dict_input.get("image", None)
    image_data = None
    if img_out is not None:
        # Expect tuple: (image, image_data_base64)
        # Only the second is used by prepare_prompt in your code
        try:
            _, image_data = img_out
        except Exception:
            # If user passed raw base64, accept it directly
            image_data = img_out

    prepare_prompt_fn = ensure_prepare_prompt_exists()
    conversation = prepare_prompt_fn(
        prompt=prompt,
        image=image_data,
        conversation_history=conversation_history,
    )

    # Inference with jittered exponential backoff
    last_exc = None
    content = ""
    for attempt in range(3):
        try:
            # Some clients accept 'timeout' as keyword; if not, it will be ignored.
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
            )
            content = response.choices[0].message.content
            break
        except Exception as e:
            last_exc = e
            traceback.print_exc()
            # 0s, ~2s, ~4s backoff with small jitter
            sleep_s = (2 ** attempt) + random.random() * 0.2
            time.sleep(sleep_s)
    else:
        # If the loop did not break (all retries failed)
        raise RuntimeError(f"Inference failed after retries: {last_exc}") from last_exc

    content = postprocess_output(content)

    if print_content:
        try:
            # Pretty print dicts/JSON-like outputs gracefully
            import json as _json
            parsed = None
            try:
                parsed = _json.loads(content)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                print(_json.dumps(parsed, ensure_ascii=False, indent=2))
            else:
                print(content)
        except Exception:
            print(content)

    return content, conversation
########################
# Robust JSON extraction
########################

def _safe_json_extract(text: str) -> dict:
    """Best-effort: pull a top-level JSON object out of a model string."""
    if not text:
        return {}
    # Try outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Regex fallback
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def _cleanup_list(
    items: List[str],
    min_words: int = 8,
    max_words: int = 16,
) -> List[str]:
    """Normalize, length-filter (by words), and dedupe while preserving order."""
    seen = set()
    out = []
    for it in items or []:
        s = (it or "").strip().strip('"').strip("'")
        s = re.sub(r"\s+", " ", s)
        if not s:
            continue
        n_words = len(s.split())
        if n_words < min_words or n_words > max_words:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

########################
# Prompt builder
########################

def build_variants_prompt(
    base_text: str,
    num_paraphrases: int = 6,
    num_contrasts: int = 6,
) -> str:
    """Create a strict JSON instruction prompt for consistent, parseable output."""
    return f"""
    You are generating task-instruction rewrites for a Vision-Language-Action (VLA) evaluation.

    BASE INSTRUCTION:
    <<<{base_text}>>>

    GOAL
    1) Paraphrases: Keep the same task and intent, but change wording and phrasing.
    2) Contrasts: Use the SAME OBJECTS/ENTITIES mentioned in BASE, but change their ORDER or RELATIONSHIPS so the task meaning is different.
    - Do NOT introduce new objects.
    - Prefer swapping source/destination, temporal order, or spatial relations over trivial negations.

    STYLE CONSTRAINTS (apply to EVERY line)
    - Single imperative sentence (no numbering, no lists).
    - Mention only objects from BASE (no new nouns).
    - No quotes, no markdown, no commentary.

    COUNTS
    - Generate exactly {num_paraphrases} Paraphrases.
    - Generate exactly {num_contrasts} Contrasts.

    OUTPUT FORMAT (STRICT JSON ONLY):
    {{
    "paraphrases": ["...", "..."],
    "contrasts": ["...", "..."]
    }}

    Return only the JSON object above, with valid JSON syntax.
    """.strip()

########################
# Core generator
########################

def generate_instruction_variants(
    base_text: str,
    num_paraphrases: int = 3,
    num_contrasts: int = 3,
    min_words: int = 8,
    max_words: int = 16,
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[Dict[str, List[str]], List[Dict]]:
    """
    Uses your existing model_infer() + prepare_prompt() (imported from your code)
    to produce two lists: paraphrases and contrasts.

    Returns:
        (variants_dict, conversation)
        variants_dict = {"paraphrases": [...], "contrasts": [...]}
    """
    # We import inside to avoid circular imports if you place this file next to your code.

    prompt = build_variants_prompt(
        base_text=base_text,
        num_paraphrases=num_paraphrases,
        num_contrasts=num_contrasts,
    )

    dict_input = {"prompt": prompt}
    content, conversation = model_infer(
        dict_input=dict_input,
        dict_config_m=dict_config_m,
        conversation_history=conversation_history,
        print_content=False,
    )

    if isinstance(content, dict):
        data = content
    else:
        data = _safe_json_extract(str(content))

    paraphrases = _cleanup_list(
        data.get("paraphrases", []), min_words=min_words, max_words=max_words
    )[:num_paraphrases]
    contrasts = _cleanup_list(
        data.get("contrasts", []), min_words=min_words, max_words=max_words
    )[:num_contrasts]

    return {"paraphrases": paraphrases, "contrasts": contrasts}, conversation


if __name__ == "__main__":
    base_text = 'turn on the stove and put the moka pot on it'
    num_paraphrases = 1
    num_contrasts = 2
    min_words = 8
    max_words = 16


    # ----------- Run generation -----------
    all_records = []
    try:
        variants, _ = generate_instruction_variants(
            base_text=base_text,
            num_paraphrases=num_paraphrases,
            num_contrasts=num_contrasts,
            min_words=min_words,
            max_words=max_words,
        )
        import pdb;pdb.set_trace()

        for typ in ["paraphrases", "contrasts"]:
            for j, t in enumerate(variants.get(typ, [])):
                all_records.append({
                    "type": typ,
                    "base": base_text,
                    "variant_index": j + 1,
                    "variant": t,
                })
    except Exception as e:
        print(f"⚠️  Error generating for '{base_text}': {e}")
