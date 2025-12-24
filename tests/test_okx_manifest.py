from pathlib import Path
import sys
import hashlib

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.okx as okx_mod


def test_okx_manifest_parsing_and_cache(tmp_path, monkeypatch):
    payload = {
        "code": "0",
        "data": [
            {
                "dateAggrType": "daily",
                "details": [
                    {
                        "date": "2025-09-01",
                        "groupDetails": [
                            {
                                "dateTs": "1756684800000",
                                "fileName": "BTC-USDT-SWAP-trades-2025-09-01.zip",
                                "url": "https://example.com/BTC-USDT-SWAP-trades-2025-09-01.zip",
                            }
                        ],
                    }
                ],
                "ts": "1766458000000",
            }
        ],
        "msg": "",
    }

    def fake_request_json(session, url, params=None, timeout=15, max_retries=5):
        return payload

    monkeypatch.setattr(okx_mod, "request_json", fake_request_json)

    client = okx_mod.OKXHistoricalClient(endpoint="https://example.com", manifest_dir=tmp_path)
    start_ms = int(pd.Timestamp("2025-09-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-09-02", tz="UTC").timestamp() * 1000)

    files = client.list_files(
        dataset="trades",
        inst_id="BTC-USDT-SWAP",
        start_ms=start_ms,
        end_ms=end_ms,
        date_aggr_type="daily",
        level=None,
        inst_type="SWAP",
        allow_remote=True,
        local_dir=tmp_path,
    )

    assert len(files) == 1
    desc = files[0]
    assert desc.url == payload["data"][0]["details"][0]["groupDetails"][0]["url"]
    assert desc.local_path == tmp_path / "trades" / "BTC-USDT-SWAP" / desc.filename

    cache = tmp_path / "manifest_1_BTC-USDT-SWAP_daily_levelna.json"
    assert cache.exists()


def test_okx_checksum_verify(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc123")
    expected = hashlib.sha256(b"abc123").hexdigest()
    okx_mod._verify_checksum(path, expected)

    with pytest.raises(ValueError):
        okx_mod._verify_checksum(path, "0" * 64)
