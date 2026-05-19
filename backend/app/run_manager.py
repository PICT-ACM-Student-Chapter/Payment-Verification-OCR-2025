import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import ID_verify
import extraction
from backend.app.schemas import RunConfig, RunCreateResponse, RunState, RunStatus, RunSummary

RUNS_ROOT = Path("tmp/runs")
ALLOWED_REGISTRATION_EXTENSIONS = {".csv", ".xlsx"}
ALLOWED_REPORT_EXTENSIONS = {".csv", ".xlsx", ".pdf"}


@dataclass
class RunRecord:
    run_id: str
    workspace: Path
    status: RunState
    stage: str
    message: str
    progress: int
    created_at: datetime
    completed_at: datetime | None = None
    summary: RunSummary | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunManager:
    def __init__(self):
        self.runs: dict[str, RunRecord] = {}
        self.lock = threading.Lock()
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    def create_run(self, registration_upload, report_uploads, config: RunConfig) -> RunCreateResponse:
        run_id = uuid.uuid4().hex
        workspace = RUNS_ROOT / run_id
        workspace.mkdir(parents=True, exist_ok=True)

        record = RunRecord(
            run_id=run_id,
            workspace=workspace,
            status="queued",
            stage="queued",
            message="Run queued",
            progress=0,
            created_at=datetime.now(timezone.utc),
        )

        with self.lock:
            self.runs[run_id] = record

        self._save_upload(workspace / "registration", registration_upload, ALLOWED_REGISTRATION_EXTENSIONS)
        reports_dir = workspace / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for upload in report_uploads:
            self._save_upload(reports_dir, upload, ALLOWED_REPORT_EXTENSIONS, use_original_name=True)

        config_payload = {
            "reg_transaction_id_column": config.regTransactionIdColumn,
            "use_fallback": config.useFallback,
            "rrn_column": config.rrnColumn,
            "amount_column": config.amountColumn,
        }
        with open(workspace / "column_config.json", "w", encoding="utf-8") as file:
            json.dump(config_payload, file)

        thread = threading.Thread(target=self._process_run, args=(run_id,), daemon=True)
        thread.start()
        return RunCreateResponse(runId=run_id, status="queued")

    def _save_upload(self, destination, upload, allowed_extensions: set[str], use_original_name: bool = False):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {upload.filename}")

        if use_original_name:
            target = Path(destination) / Path(upload.filename).name
        else:
            target = Path(f"{destination}{suffix}")

        with open(target, "wb") as file:
            shutil.copyfileobj(upload.file, file)

    def _update_run(
        self,
        run_id: str,
        *,
        status: RunState | None = None,
        stage: str | None = None,
        message: str | None = None,
        progress: int | None = None,
        summary: RunSummary | None = None,
        error: str | None = None,
        completed: bool = False,
    ):
        record = self.runs[run_id]
        with record.lock:
            if status is not None:
                record.status = status
            if stage is not None:
                record.stage = stage
            if message is not None:
                record.message = message
            if progress is not None:
                record.progress = progress
            if summary is not None:
                record.summary = summary
            if error is not None:
                record.error = error
            if completed:
                record.completed_at = datetime.now(timezone.utc)

    def _process_run(self, run_id: str):
        record = self.runs[run_id]
        workspace = record.workspace
        config_path = workspace / "column_config.json"
        registration_path = next((workspace / "registration").parent.glob("registration.*"))
        report_paths = sorted(str(path) for path in (workspace / "reports").iterdir() if path.is_file())
        processed_path = workspace / "processed_transactions.csv"
        verified_path = workspace / "verified_transactions.csv"
        verified_db_path = workspace / "verified_ID.csv"

        try:
            self._update_run(
                run_id,
                status="processing_uploads",
                stage="processing_uploads",
                message="Files received. Preparing run workspace.",
                progress=10,
            )

            def on_extract_progress(current: int, total: int):
                base = 20
                span = 45
                progress = base if total == 0 else base + int((current / total) * span)
                self._update_run(
                    run_id,
                    status="extracting_ids",
                    stage="extracting_ids",
                    message=f"Extracting transaction IDs from screenshot {current} of {total}.",
                    progress=min(progress, 65),
                )

            extraction.run_extraction(
                reg_path=str(registration_path),
                output_path=str(processed_path),
                config_path=str(config_path),
                model_path="model.pt",
                progress_callback=on_extract_progress,
            )

            self._update_run(
                run_id,
                status="verifying",
                stage="verifying",
                message="Matching extracted IDs against transaction reports.",
                progress=75,
            )

            df = ID_verify.run_verification(
                extracted_data_path=str(processed_path),
                report_paths=report_paths,
                output_path=str(verified_path),
                verified_db_path=str(verified_db_path),
                config_path=str(config_path),
            )

            summary = build_summary(df)
            self._update_run(
                run_id,
                status="completed",
                stage="completed",
                message="Verification completed.",
                progress=100,
                summary=summary,
                completed=True,
            )
        except Exception as exc:
            self._update_run(
                run_id,
                status="failed",
                stage="failed",
                message="Verification failed.",
                progress=100,
                error=str(exc),
                completed=True,
            )

    def get_status(self, run_id: str) -> RunStatus:
        record = self._require_run(run_id)
        return RunStatus(
            runId=record.run_id,
            status=record.status,
            stage=record.stage,
            message=record.message,
            progress=record.progress,
            createdAt=record.created_at,
            completedAt=record.completed_at,
            summary=record.summary,
            error=record.error,
        )

    def get_results(self, run_id: str, page: int, page_size: int, verification: str | None):
        record = self._require_run(run_id)
        results_path = record.workspace / "verified_transactions.csv"
        if not results_path.exists():
            raise FileNotFoundError("Run results are not available yet.")

        df = pd.read_csv(results_path, dtype=str).fillna("")
        if verification:
            df = df[df["Verification"] == verification]

        total = len(df)
        start = max(page - 1, 0) * page_size
        end = start + page_size
        rows = df.iloc[start:end].to_dict(orient="records")
        return total, rows

    def get_download_path(self, run_id: str) -> Path:
        record = self._require_run(run_id)
        results_path = record.workspace / "verified_transactions.csv"
        if not results_path.exists():
            raise FileNotFoundError("Run results are not available yet.")
        return results_path

    def delete_run(self, run_id: str):
        record = self._require_run(run_id)
        shutil.rmtree(record.workspace, ignore_errors=True)
        with self.lock:
            del self.runs[run_id]

    def _require_run(self, run_id: str) -> RunRecord:
        with self.lock:
            if run_id not in self.runs:
                raise KeyError(run_id)
            return self.runs[run_id]


def build_summary(df: pd.DataFrame) -> RunSummary:
    counts = df["Verification"].value_counts(dropna=False)
    return RunSummary(
        totalRecords=len(df),
        verified=int(counts.get("Verified", 0)),
        notVerified=int(counts.get("Not Verified", 0)),
        noIdExtracted=int(counts.get("No ID extracted", 0)),
        duplicates=int(counts.get("Duplicate", 0)),
        registrationDuplicates=int(counts.get("Registration Duplicate", 0)),
        amountMismatch=int(counts.get("Amount mismatch", 0)),
    )


run_manager = RunManager()
