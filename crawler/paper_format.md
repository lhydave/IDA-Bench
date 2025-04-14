# Paper record format

Each paper record should be a json item with the following fields:
- `title`: The title of the paper.
- `where_published`: The venue where the paper was published (e.g., Nature, arXiv, AER, etc.).
- `when_published`: The date when the paper was published (e.g., 2023-10-01).
- `is_open_access`: A boolean indicating whether the paper is open access or not.
- `paper_url`: The URL of the paper, which could access full text (e.g., PDF/html).
- `subject`: The subject of the paper (e.g., economics, finance, etc.).
- `content_path`: The path to the content of the paper, which should be a string to a text file.

Some optional fields:
- `code_availability`: The code availability section of the paper, if available.
- `data_availability`: The data availability section of the paper, if available.
- `code_path`: The path to the concatenated code of the paper, which should be a string to a text file.
- `metrics`: The metrics of the paper, such as citations, downloads, etc.

## Example

```json
{
    "title": "A Simple Model of the Phillips Curve in a New Keynesian Framework",

    "where_published": "AER",

    "when_published": "2018-11-01",

    "is_open_access": true,

    "paper_url": "https://www.aeaweb.org/articles?id=10.1257/aer.20181100",

    "subject": ["economics", "finance"],

    "content_path": "papers/2018-11-01/aer_20181100.txt",

    "code_availability": "The source code is available at GitHub repo: https://github.com/xxx",

    "data_availability": "The data used in this paper is available at: https://zenodo.org/xxx",

    "code_path": "papers/2018-11-01/aer_20181100_code.txt",

    "metrics": {
        "citations": 100,
        "downloads": 500,
        "views": 1000
    }
}
```