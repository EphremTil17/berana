from typing import Any
from unittest.mock import patch

import pytest
import typer

from modules.cli.runtime import execute_pipeline


def faulty_pipeline(**kwargs: Any) -> Any:
    """Mock pipeline that always fails."""
    raise ValueError("Simulated pipeline failure.")


def test_cli_failure_exit_code():
    """Ensure a pipeline command correctly exits with code 1 and formats the error properly via the wrapper."""
    with patch("modules.cli.runtime.ensure_pdf_exists") as mock_ensure:
        # Bypass file existence check
        mock_ensure.return_value = "fake_path.pdf"

        with pytest.raises(typer.Exit) as exc_info:
            execute_pipeline(
                pdf_path="fake_path.pdf",
                pipeline_fn=faulty_pipeline,
                context_label="Test Failure",
                success_msg="Should not print",
            )

        assert exc_info.value.exit_code == 1
