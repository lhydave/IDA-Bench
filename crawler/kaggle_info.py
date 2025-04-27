from dataclasses import dataclass, asdict
from typing import Literal
import json


@dataclass
class DatasetInfo:
    url: str
    name: str
    id: str
    type: Literal["dataset", "competition"]
    description: str
    date: str
    filename_list: list[str]
    path: str | None = None  # local path if available

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetInfo":
        """Create a DatasetInfo instance from a JSON string."""
        return cls(**json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert DatasetInfo instance to a dictionary."""
        result = {}
        for k, v in asdict(self).items():
            # Skip None values except for fields that can be None
            if v is not None or k in ["dataset_path"]:
                result[k] = v
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert DatasetInfo instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class NotebookInfo:
    url: str
    title: str
    id: str
    date: str
    votes: int
    copy_and_edit: int
    views: int
    comments: int
    runtime: int  # in seconds
    input_size: float  # in B
    input: list[DatasetInfo]  # list of dataset json
    prize: str | None = None  # if available
    path: str | None = None  # local path if available

    @classmethod
    def from_json(cls, json_str: str) -> "NotebookInfo":
        """Create a NotebookInfo instance from a JSON string."""
        return cls(**json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert NotebookInfo instance to a dictionary."""
        result = {}
        for k, v in asdict(self).items():
            # Special handling for input list
            # Skip None values except for fields that can be None
            if v is not None or k in ["prize", "path"]:
                result[k] = v
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert NotebookInfo instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
