import json

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.run_manager import run_manager
from backend.app.schemas import ResultsResponse, RunConfig, RunCreateResponse, RunStatus

app = FastAPI(title="Payment Verification OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/runs", response_model=RunCreateResponse)
async def create_run(
    registration_file: UploadFile = File(...),
    report_files: list[UploadFile] = File(...),
    config: str = Form(...),
):
    try:
        config_model = RunConfig.model_validate(json.loads(config))
        return run_manager.create_run(registration_file, report_files, config_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid config JSON.") from exc


@app.get("/api/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str):
    try:
        return run_manager.get_status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@app.get("/api/runs/{run_id}/results", response_model=ResultsResponse)
def get_results(
    run_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=250),
    verification: str | None = Query(None),
):
    try:
        total, rows = run_manager.get_results(run_id, page, page_size, verification)
        return ResultsResponse(runId=run_id, total=total, page=page, pageSize=page_size, rows=rows)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/download")
def download_results(run_id: str):
    try:
        path = run_manager.get_download_path(run_id)
        return FileResponse(path, media_type="text/csv", filename="verified_transactions.csv")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    try:
        run_manager.delete_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
