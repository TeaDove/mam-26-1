from pydantic import BaseModel


def flatten(model: BaseModel) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in model.model_dump().items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}.{inner_key}"] = inner_value
        elif isinstance(value, list):
            flat[key] = ", ".join(str(item) for item in value[:8])
        else:
            flat[key] = value
    return flat


def markdown_table(title: str, columns: dict[str, BaseModel]) -> str:
    flat = {name: flatten(model) for name, model in columns.items()}
    keys = list(next(iter(flat.values())).keys())
    lines = [f"## {title}", "", "| metric | " + " | ".join(columns) + " |"]
    lines.append("|" + " --- |" * (len(columns) + 1))
    for key in keys:
        cells = [_format(flat[name].get(key)) for name in columns]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _format(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")
