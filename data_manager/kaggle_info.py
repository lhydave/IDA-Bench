from dataclasses import dataclass, asdict
from typing import Literal
import json


@dataclass
class DatasetInfo:
    url: str
    name: str
    type: Literal["dataset", "competition"]
    description: str
    date: str
    contain_time_series: bool
    filename_list: list[str]
    path: str | None = None  # local path if available

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetInfo":
        """Create a DatasetInfo instance from a JSON string."""
        return cls(**json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert DatasetInfo instance to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert DatasetInfo instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class CodeInfo:
    num_pivot_table: int
    num_groupby: int
    num_apply: int
    num_def: int
    num_for: int
    num_and: int
    num_or: int
    num_merge: int
    num_concat: int
    num_join: int
    num_agg: int
    num_python_cell: int  # number of python cells
    import_list: list[str]
    file_size: int  # in bytes
    pure_code_size: int  # in bytes


@dataclass
class NotebookInfo:
    url: str
    title: str
    date: str
    votes: int
    copy_and_edit: int
    views: int
    comments: int
    runtime: int  # in seconds
    input_size: float  # in B
    input: list[str]  # list of input dataset ID
    prize: str | None = None  # if available
    path: str | None = None  # local path if available
    code_info: CodeInfo | None = None  # code info json

    @classmethod
    def from_json(cls, json_str: str) -> "NotebookInfo":
        """Create a NotebookInfo instance from a JSON string."""
        return cls(**json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert NotebookInfo instance to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert NotebookInfo instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
