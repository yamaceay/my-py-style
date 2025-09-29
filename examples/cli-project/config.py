from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    app_name: str
    output_dir: Path
    verbose: bool
    config_file: str | None

def create_config(
    app_name: str,
    output_dir: str,
    verbose: bool = False,
    config_file: str | None = None
) -> AppConfig:
    return AppConfig(
        app_name=app_name,
        output_dir=Path(output_dir),
        verbose=verbose,
        config_file=config_file
    )
