import json
import shutil
from pathlib import Path
from typing import Any, Union

ROOT_PATH = Path(__file__).parent.parent.parent


def read_json(file_path: Union[str, Path]) -> Any:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with file_path.open('r', encoding='utf-8') as file:
        return json.load(file)


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def read_lines(file_path: Union[str, Path]) -> list[str]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    with file_path.open('r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def write_lines(lines: list[str], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')


def safe_file_copy(source: Union[str, Path], destination: Union[str, Path]) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
