import gzip
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.cli.inspect_bybit_l2 import inspect_bybit_l2
from src.data.bybit_manual import scan_orderbook_payloads, sniff_file


def _setup_root(tmp_path: Path, fixture_name: str) -> Path:
    fixture = Path(__file__).resolve().parent / "fixtures" / "bybit_l2_small" / fixture_name
    root = tmp_path / "bybit_root"
    folder = root / "2025-12-01_BTCUSDT_ob200.data"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, folder / "orderbook.data")
    return root


@pytest.mark.parametrize(
    ("fixture_name", "expected_scale"),
    [
        ("orderbook_seconds.jsonl", "seconds"),
        ("orderbook_ms.jsonl", "milliseconds"),
    ],
)
def test_bybit_l2_inspector_ts_scale(tmp_path, fixture_name, expected_scale):
    root = _setup_root(tmp_path, fixture_name)
    result = inspect_bybit_l2(
        root=root,
        symbol="BTCUSDT",
        date="2025-12-01",
        level=200,
        output_dir=tmp_path,
    )
    assert (tmp_path / "bybit_l2_inspect.json").exists()
    assert result["files"]
    file_info = result["files"][0]
    assert file_info["format"] == "jsonl"
    assert file_info["events_in_window"] > 0
    assert file_info["ts_scale_counts"][expected_scale] > 0
    assert file_info["min_ts_ms"] == 1764547200000


def test_bybit_orderbook_discovery_prefers_largest(tmp_path):
    root = tmp_path / "bybit_root"
    folder = root / "2025-12-04_BTCUSDT_ob200.data"
    folder.mkdir(parents=True, exist_ok=True)
    line = (
        '{"type":"snapshot","ts":1764547200,"data":{"s":"BTCUSDT","seq":1,"u":1,"a":[["101","1"]],"b":[["100","1"]]}}'
    )
    small = folder / "small.jsonl"
    big = folder / "big.jsonl"
    small.write_text(line + "\n", encoding="utf-8")
    big.write_text((line + "\n") * 10, encoding="utf-8")

    payloads = scan_orderbook_payloads(root, "BTCUSDT")
    assert payloads
    assert payloads[0]["selected"] == big


def test_bybit_l2_sniff_gzip_zip(tmp_path):
    raw = tmp_path / "orderbook.jsonl"
    raw.write_text(
        (
            '{"type":"snapshot","ts":1764547200,"data":{"s":"BTCUSDT","seq":1,"u":1,"a":[["101","1"]],"b":[["100","1"]]}}'
            + "\n"
            + '{"type":"delta","ts":1764547201,"data":{"s":"BTCUSDT","seq":2,"u":2,"a":[["102","1"]],"b":[["99","1"]]}}'
            + "\n"
        ),
        encoding="utf-8",
    )

    gz_path = tmp_path / "orderbook.jsonl.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(raw.read_bytes())

    zip_path = tmp_path / "orderbook.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.write(raw, arcname="orderbook.jsonl")

    sniff = sniff_file(gz_path)
    assert sniff.container == "gzip"
    assert sniff.data_format == "jsonl"

    sniff = sniff_file(zip_path)
    assert sniff.container == "zip"
    assert sniff.data_format == "jsonl"
