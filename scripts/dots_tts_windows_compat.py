"""Make optional upstream text-normalizer imports lazy on native Windows.

Cloning uses upstream's default normalize_text=False. Pynini is only needed
when text normalization is explicitly requested; it has no Windows wheel.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    spec = importlib.util.find_spec("dots_tts")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install dots.tts before applying Windows compatibility.")
    target = Path(next(iter(spec.submodule_search_locations))) / "utils" / "text.py"
    source = target.read_text(encoding="utf-8")
    for language, alias in (("chinese", "ZhNormalizer"), ("english", "EnNormalizer")):
        line = f"from tn.{language}.normalizer import Normalizer as {alias}"
        function = f"def get_{language}_text_normalizer() -> {alias}:\n"
        lazy = f"{function}    {line}\n"
        if lazy in source:
            continue
        if f"\n{line}\n" not in source or function not in source:
            raise RuntimeError(f"Unexpected dots.tts normalizer source: {target}")
        source = source.replace(f"\n{line}\n", "\n", 1)
        source = source.replace(function, lazy, 1)
    target.write_text(source, encoding="utf-8")
    print(f"Optional normalization imports are lazy: {target}")


if __name__ == "__main__":
    main()
