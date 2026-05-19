import pandas as pd

import ID_verify
import extraction
from backend.app.run_manager import build_summary


def test_detect_screenshot_column_accepts_plural():
    frame = pd.DataFrame({"screenshots": ["https://example.com"]})
    assert extraction.detect_screenshot_column(frame) == "screenshots"


def test_registration_duplicates_marked():
    frame = pd.DataFrame(
        {
            "transactionId": ["ABC123456789", "ABC123456789", "DIFF123456789"],
            "extracted_transaction_id": ["ABC123456789", "ABC123456789", "DIFF123456789"],
            "Verification": ["Verified", "Verified", "Verified"],
        }
    )

    result = ID_verify.check_registration_duplicates(frame, {"reg_transaction_id_column": "transactionId"})
    assert list(result["Verification"]) == ["Registration Duplicate", "Registration Duplicate", "Verified"]


def test_build_summary_counts_expected_statuses():
    frame = pd.DataFrame(
        {
            "Verification": [
                "Verified",
                "Not Verified",
                "No ID extracted",
                "Duplicate",
                "Registration Duplicate",
                "Amount mismatch",
            ]
        }
    )

    summary = build_summary(frame)
    assert summary.totalRecords == 6
    assert summary.verified == 1
    assert summary.notVerified == 1
    assert summary.noIdExtracted == 1
    assert summary.duplicates == 1
    assert summary.registrationDuplicates == 1
    assert summary.amountMismatch == 1
