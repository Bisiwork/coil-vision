# src/tablegen/gpt_table_parser.py

import os
import re
import json
from openai import OpenAI
from typing import Any, Dict, List, Union
from pydantic import TypeAdapter
from .row_schema import SpringProgramRow

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SPRING_ROW_LIST_ADAPTER = TypeAdapter(List[SpringProgramRow])

# Mappatura tra chiavi GPT e attributi SpringProgramRow
_FIELD_MAP = {
    "feed": "feed",
    "cumulative_feed": "feed_abs",
    "diameter": "diameter",
    "rotation": "rotation",
    "vertical_pitch": "v_pitch",
    "horizontal_pitch": "h_pitch",
    "function_codes": "funct",
    "param": "param",
    "speed": "speed",
}

def _to_float(val: Any) -> Union[float, None]:
    """Estrai un numero float da stringhe tipo '1,808' o '+0,05'."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    # trova il primo match di un numero (con virgola o punto)
    m = re.search(r"[+-]?\d+[.,]?\d*", val)
    if not m:
        return None
    return float(m.group(0).replace(",", "."))

def _to_int(val: Any) -> Union[int, None]:
    """Estrai un intero da stringhe con cifre."""
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if not isinstance(val, str):
        return None
    m = re.search(r"\d+", val)
    if not m:
        return None
    return int(m.group(0))

def _normalize_raw_row(raw: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Converti e pulisci un dict GPT in un dict adatto a SpringProgramRow."""
    norm: Dict[str, Any] = {"step": step}

    # Mappa e pulisci i campi principali
    for gpt_key, our_key in _FIELD_MAP.items():
        v = raw.get(gpt_key)
        if our_key in ("feed", "feed_abs", "diameter", "rotation", "v_pitch", "h_pitch"):
            norm[our_key] = _to_float(v)
        elif our_key in ("funct", "param", "speed"):
            norm[our_key] = _to_int(v)
        else:
            # se per caso altri campi numerici
            norm[our_key] = v

    # Mandrel positions: estrai y e z e parsali float
    mp = raw.get("mandrel_positions", {})
    norm["mand_y"] = _to_float(mp.get("y"))
    norm["mand_z"] = _to_float(mp.get("z"))

    return norm

def generate_table_from_ocr_text(ocr_text: str) -> List[SpringProgramRow]:
    prompt = f"""
You are a spring machine programmer copilot.
Given this technical drawing text, return **only** JSON (no explanations, no fences), either:
- An object {{ "spring_segments": [ ... ] }}.

Each row must include keys:
  feed, cumulative_feed, diameter, rotation, vertical_pitch, horizontal_pitch,
  mandrel_positions ({{y,z}}), function_codes, param, speed.

Here is the OCR text:
'''{ocr_text}'''
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=2048,
        response_format={"type":"json_object"}
    )

    content = response.choices[0].message.content
    print("📋 Raw GPT response:", content)

    # Decodifica JSON
    if isinstance(content, str):
        text = content.strip().strip("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error:", e)
            return []
    else:
        data = content

    # Estrai lista di segmenti
    if isinstance(data, dict) and "spring_segments" in data:
        segments = data["spring_segments"]
    elif isinstance(data, list):
        segments = data
    else:
        print("❌ Unexpected JSON structure:", data)
        return []

    # Normalizza ogni riga e assegna step
    normalized_rows = [
        _normalize_raw_row(raw_row, step=i+1)
        for i, raw_row in enumerate(segments)
    ]

    # Valida con Pydantic v2 (TypeAdapter)
    try:
        return SPRING_ROW_LIST_ADAPTER.validate_python(normalized_rows)
    except Exception as e:
        print("❌ Validation error:", e)
        print("Normalized rows:", normalized_rows)
        return []
