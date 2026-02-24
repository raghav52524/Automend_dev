"""
Unit tests for Track B Combiner (Generative Architect).

Tests the combine_track_b function that concatenates JSONL files
from Datasets 3-6 (StackOverflow, Synthetic, Glaive, The Stack).

Run with: pytest tests/test_combiner_track_b.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestCombineTrackB:
    """Unit tests for Track B combiner."""

    @pytest.mark.unit
    def test_combine_track_b_concatenates_records(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verifies record counts sum correctly after concatenation."""
        from combiner_track_b import combine

        # Count expected records
        expected_total = 0
        for file_path in sample_track_b_files:
            with open(file_path, "r", encoding="utf-8") as f:
                expected_total += sum(1 for line in f if line.strip())

        interim_dir = sample_track_b_files[0].parent

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_B_combined.jsonl")

        total, by_source = combine.combine_track_b()

        assert total == expected_total

    @pytest.mark.unit
    def test_combine_track_b_adds_source_column(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verifies source_dataset field is added to each record."""
        from combiner_track_b import combine

        interim_dir = sample_track_b_files[0].parent
        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        # Read output and check source_dataset field
        sources_found = set()
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "source_dataset" in record
                sources_found.add(record["source_dataset"])

        expected_sources = {p.stem for p in sample_track_b_files}
        assert sources_found == expected_sources

    @pytest.mark.unit
    def test_combine_track_b_preserves_json_structure(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verifies each output line is valid JSON with messages key."""
        from combiner_track_b import combine

        interim_dir = sample_track_b_files[0].parent
        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    assert "messages" in record, f"Line {line_num} missing 'messages' key"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {line_num} is not valid JSON: {e}")

    @pytest.mark.unit
    def test_combine_track_b_handles_missing_file(
        self, sample_jsonl_ds3, sample_jsonl_ds4, tmp_interim_dir, tmp_processed_dir, monkeypatch
    ):
        """Verifies graceful handling when some input files are missing."""
        from combiner_track_b import combine

        nonexistent = tmp_interim_dir / "nonexistent.jsonl"
        input_files = [sample_jsonl_ds3, sample_jsonl_ds4, nonexistent]

        monkeypatch.setattr(combine, "INTERIM_DIR", tmp_interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", input_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", tmp_processed_dir / "track_B_combined.jsonl")

        # Should still work with available files
        total, by_source = combine.combine_track_b()

        # Count expected from existing files
        expected = 0
        for f in [sample_jsonl_ds3, sample_jsonl_ds4]:
            with open(f, "r", encoding="utf-8") as fh:
                expected += sum(1 for line in fh if line.strip())

        assert total == expected

    @pytest.mark.unit
    def test_combine_track_b_handles_malformed_json(
        self, malformed_jsonl_file, tmp_processed_dir, monkeypatch
    ):
        """Verifies combiner skips malformed JSON lines and continues."""
        from combiner_track_b import combine

        interim_dir = malformed_jsonl_file.parent
        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", [malformed_jsonl_file])
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        # Should not raise, but skip malformed lines
        total, by_source = combine.combine_track_b()

        # Only 2 valid JSON lines in malformed_jsonl_file fixture
        assert total == 2

    @pytest.mark.unit
    def test_combine_track_b_output_is_valid_jsonl(
        self, sample_track_b_files, tmp_processed_dir, monkeypatch
    ):
        """Verifies output file is readable line by line as valid JSONL."""
        from combiner_track_b import combine

        interim_dir = sample_track_b_files[0].parent
        output_path = tmp_processed_dir / "track_B_combined.jsonl"

        monkeypatch.setattr(combine, "INTERIM_DIR", interim_dir)
        monkeypatch.setattr(combine, "PROCESSED_DIR", tmp_processed_dir)
        monkeypatch.setattr(combine, "INPUT_FILES", sample_track_b_files)
        monkeypatch.setattr(combine, "OUTPUT_FILE", output_path)

        combine.combine_track_b()

        # Verify file exists and all lines are valid JSON
        assert output_path.exists()

        valid_lines = 0
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)  # Will raise if invalid
                    valid_lines += 1

        assert valid_lines > 0
