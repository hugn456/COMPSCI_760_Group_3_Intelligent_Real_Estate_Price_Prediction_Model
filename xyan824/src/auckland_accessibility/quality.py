from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import utc_now


def write_quality_report(
    stem: str,
    metrics: dict[str, Any],
    reports_dir: Path,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": utc_now(), **metrics}
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"generated_{stem}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")

    lines = [f"# {stem.replace('_', ' ').title()}", ""]
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.extend((f"## {key}", ""))
            for child_key, child_value in value.items():
                lines.append(f"- `{child_key}`: {child_value}")
            lines.append("")
        elif isinstance(value, list):
            lines.extend((f"## {key}", ""))
            lines.extend(f"- {item}" for item in value)
            lines.append("")
        else:
            lines.append(f"- `{key}`: {value}")
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path

