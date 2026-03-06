from modules.ocr_training.surya_inspect import (
    build_batch_geometry,
    build_truncation_report,
    parse_int_csv,
    select_rows_for_inspection,
    summarize_lengths,
)


def test_parse_int_csv_returns_sorted_unique_values():
    assert parse_int_csv("4,2,2,1") == [1, 2, 4]


def test_select_rows_for_inspection_is_deterministic():
    rows = [{"image": f"/tmp/{index}.png", "text": str(index)} for index in range(10)]

    first = select_rows_for_inspection(rows, sample_size=4, seed=42)
    second = select_rows_for_inspection(rows, sample_size=4, seed=42)

    assert first == second
    assert len(first) == 4


def test_summarize_lengths_reports_distribution():
    summary = summarize_lengths([10, 20, 30, 40, 50])

    assert summary["count"] == 5
    assert summary["min"] == 10
    assert summary["p50"] == 30
    assert summary["max"] == 50


def test_build_truncation_report_counts_clipped_rows():
    report = build_truncation_report([100, 200, 300], max_sequence_lengths=[128, 256])

    assert report == [
        {"max_sequence_length": 128, "clipped_rows": 2, "clipped_rate": 2 / 3},
        {"max_sequence_length": 256, "clipped_rows": 1, "clipped_rate": 1 / 3},
    ]


def test_build_batch_geometry_derives_effective_batches_and_steps():
    rows = build_batch_geometry(
        total_rows=100,
        per_device_batch_sizes=[1, 2],
        gradient_accumulation_steps=[2, 4],
    )

    assert rows == [
        {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 2,
            "optimizer_steps_per_epoch": 50,
        },
        {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 4,
            "optimizer_steps_per_epoch": 25,
        },
        {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 4,
            "optimizer_steps_per_epoch": 25,
        },
        {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "optimizer_steps_per_epoch": 13,
        },
    ]
