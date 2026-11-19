"""Dataset definitions for structured (SQL) querying, loaded from config/datasets.yaml."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

ColumnType = Literal["TEXT", "INTEGER", "REAL", "DATE"]
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class ColumnSpec(BaseModel):
    type: ColumnType
    description: str = ""
    categorical: bool = False  # list its distinct values in the LLM prompt


class DatasetSpec(BaseModel):
    department: str  # RBAC: who may query it
    source: Path
    description: str
    columns: dict[str, ColumnSpec] = Field(min_length=1)

    @field_validator("columns")
    @classmethod
    def _safe_column_names(cls, columns: dict[str, ColumnSpec]) -> dict[str, ColumnSpec]:
        # Names are interpolated into CREATE TABLE, so only plain identifiers.
        bad = [name for name in columns if not _IDENTIFIER.match(name)]
        if bad:
            raise ValueError(f"invalid column names: {bad}")
        return columns


class DatasetsFile(BaseModel):
    datasets: dict[str, DatasetSpec]

    @field_validator("datasets")
    @classmethod
    def _safe_table_names(cls, datasets: dict[str, DatasetSpec]) -> dict[str, DatasetSpec]:
        bad = [name for name in datasets if not _IDENTIFIER.match(name)]
        if bad:
            raise ValueError(f"invalid table names: {bad}")
        return datasets


def load_datasets(path: Path) -> dict[str, DatasetSpec]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return DatasetsFile.model_validate(raw).datasets
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise RuntimeError(f"Invalid datasets file {path}: {exc}") from exc
