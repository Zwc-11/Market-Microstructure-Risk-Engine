from pathlib import Path
import sys
import gzip
import json
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx_manual import sniff_file


def test_okx_manual_sniff_formats(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    sniff = sniff_file(csv_path)
    assert sniff.container == "raw"
    assert sniff.data_format == "csv"

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
    sniff = sniff_file(json_path)
    assert sniff.data_format == "json"

    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    sniff = sniff_file(jsonl_path)
    assert sniff.data_format == "jsonl"

    gz_path = tmp_path / "sample.csv.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"a,b\n1,2\n")
    sniff = sniff_file(gz_path)
    assert sniff.container == "gzip"
    assert sniff.data_format == "csv"

    zip_path = tmp_path / "sample.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("inner.csv", "a,b\n1,2\n")
    sniff = sniff_file(zip_path)
    assert sniff.container == "zip"
    assert sniff.data_format == "csv"
