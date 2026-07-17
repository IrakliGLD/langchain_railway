"""F8 contract tests for the production privacy-log canary scanner."""

from __future__ import annotations

import json

from scripts.scan_privacy_log_canaries import CANARIES, main, scan_paths


def test_clean_log_canary_scan_returns_content_free_evidence(tmp_path):
    log_file = tmp_path / "railway.log"
    log_file.write_text("request completed status=200\n", encoding="utf-8")

    report = scan_paths([log_file], max_file_bytes=1024)

    assert report == {
        "schema_version": 1,
        "clean": True,
        "files_scanned": 1,
        "bytes_scanned": log_file.stat().st_size,
        "finding_counts": {},
    }


def test_leak_scan_fails_without_echoing_canary_values(tmp_path, capsys):
    log_file = tmp_path / "edge.log"
    log_file.write_bytes(CANARIES["email"] + b"\n" + CANARIES["uuid"])

    assert main([str(log_file)]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)

    assert report["clean"] is False
    assert report["finding_counts"] == {"email": 1, "uuid": 1}
    for canary in CANARIES.values():
        assert canary.decode("ascii") not in output


def test_scanner_rejects_non_file_inputs(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["clean"] is False
    assert "scan_error" in report
