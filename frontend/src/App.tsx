import { useEffect, useState, startTransition } from "react";

import { StepRail } from "./components/StepRail";
import { SummaryCards } from "./components/SummaryCards";
import { readTabularPreview } from "./lib/filePreview";
import type { PreviewData, ResultsResponse, RunConfig, RunStatus, WizardStep } from "./lib/types";

const verificationFilters = [
  "All",
  "Verified",
  "Not Verified",
  "No ID extracted",
  "Registration Duplicate",
  "Duplicate",
  "Amount mismatch"
] as const;

function App() {
  const [currentStep, setCurrentStep] = useState<WizardStep>("upload");
  const [registrationFile, setRegistrationFile] = useState<File | null>(null);
  const [reportFiles, setReportFiles] = useState<File[]>([]);
  const [registrationPreview, setRegistrationPreview] = useState<PreviewData | null>(null);
  const [reportPreview, setReportPreview] = useState<PreviewData | null>(null);
  const [config, setConfig] = useState<RunConfig>({
    regTransactionIdColumn: null,
    useFallback: true,
    rrnColumn: null,
    amountColumn: null
  });
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [resultsPage, setResultsPage] = useState(1);
  const [resultsFilter, setResultsFilter] = useState<(typeof verificationFilters)[number]>("All");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!registrationFile) {
      setRegistrationPreview(null);
      return;
    }

    readTabularPreview(registrationFile)
      .then((preview) => {
        setRegistrationPreview(preview);
        if (preview && preview.columns.length > 0) {
          setConfig((current) => ({
            ...current,
            regTransactionIdColumn: current.regTransactionIdColumn ?? preview.columns[0]
          }));
        }
      })
      .catch(() => setErrorMessage("Failed to preview the registration sheet."));
  }, [registrationFile]);

  useEffect(() => {
    const firstTabularReport = reportFiles.find((file) => {
      const lower = file.name.toLowerCase();
      return lower.endsWith(".csv") || lower.endsWith(".xlsx");
    });

    if (!firstTabularReport) {
      setReportPreview(null);
      return;
    }

    readTabularPreview(firstTabularReport)
      .then((preview) => {
        setReportPreview(preview);
        if (preview && preview.columns.length > 0) {
          setConfig((current) => ({
            ...current,
            rrnColumn: current.rrnColumn ?? preview.columns[0],
            amountColumn: current.amountColumn ?? preview.columns[Math.min(1, preview.columns.length - 1)] ?? null
          }));
        }
      })
      .catch(() => setErrorMessage("Failed to preview the transaction report."));
  }, [reportFiles]);

  useEffect(() => {
    if (!runStatus || (runStatus.status !== "queued" && runStatus.status !== "processing_uploads" && runStatus.status !== "extracting_ids" && runStatus.status !== "verifying")) {
      return;
    }

    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/runs/${runStatus.runId}`);
      if (!response.ok) {
        return;
      }

      const data: RunStatus = await response.json();
      setRunStatus(data);
      if (data.status === "completed") {
        setCurrentStep("results");
      }
    }, 2000);

    return () => window.clearInterval(timer);
  }, [runStatus]);

  useEffect(() => {
    if (!runStatus || runStatus.status !== "completed") {
      return;
    }

    void loadResults(runStatus.runId, resultsPage, resultsFilter);
  }, [runStatus, resultsPage, resultsFilter]);

  async function loadResults(runId: string, page: number, filter: string) {
    const params = new URLSearchParams({
      page: String(page),
      page_size: "25"
    });

    if (filter !== "All") {
      params.set("verification", filter);
    }

    const response = await fetch(`/api/runs/${runId}/results?${params.toString()}`);
    if (!response.ok) {
      setErrorMessage("Failed to load results.");
      return;
    }

    const data: ResultsResponse = await response.json();
    setResults(data);
  }

  function validateUploadStep() {
    if (!registrationFile) {
      setErrorMessage("Upload a registration CSV or XLSX file.");
      return false;
    }
    if (reportFiles.length === 0) {
      setErrorMessage("Upload at least one transaction report.");
      return false;
    }
    setErrorMessage(null);
    return true;
  }

  function validateMappingStep() {
    if (!config.regTransactionIdColumn) {
      setErrorMessage("Choose the registration transaction ID column.");
      return false;
    }
    if (reportPreview && (!config.rrnColumn || !config.amountColumn)) {
      setErrorMessage("Choose the report RRN and amount columns.");
      return false;
    }
    setErrorMessage(null);
    return true;
  }

  async function submitRun() {
    if (!registrationFile || reportFiles.length === 0) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    const form = new FormData();
    form.append("registration_file", registrationFile);
    reportFiles.forEach((file) => form.append("report_files", file));
    form.append("config", JSON.stringify(config));

    const response = await fetch("/api/runs", {
      method: "POST",
      body: form
    });

    setIsSubmitting(false);

    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: "Run failed to start." }));
      setErrorMessage(payload.detail ?? "Run failed to start.");
      return;
    }

    const payload: { runId: string; status: RunStatus["status"] } = await response.json();
    setRunStatus({
      runId: payload.runId,
      status: payload.status,
      stage: "queued",
      message: "Run queued",
      progress: 0,
      createdAt: new Date().toISOString(),
      completedAt: null,
      summary: null,
      error: null
    });
    setCurrentStep("processing");
  }

  function resetFlow() {
    setCurrentStep("upload");
    setRegistrationFile(null);
    setReportFiles([]);
    setRegistrationPreview(null);
    setReportPreview(null);
    setRunStatus(null);
    setResults(null);
    setResultsPage(1);
    setResultsFilter("All");
    setErrorMessage(null);
    setConfig({
      regTransactionIdColumn: null,
      useFallback: true,
      rrnColumn: null,
      amountColumn: null
    });
  }

  function moveToMapping() {
    if (validateUploadStep()) {
      setCurrentStep("mapping");
    }
  }

  function moveToProcessing() {
    if (validateMappingStep()) {
      void submitRun();
    }
  }

  function setFilter(nextFilter: (typeof verificationFilters)[number]) {
    startTransition(() => {
      setResultsFilter(nextFilter);
      setResultsPage(1);
    });
  }

  return (
    <div className="app-shell">
      <div className="background-grid" />
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <main className="layout">
        <section className="hero glass-panel-strong neon-glow">
          <span className="hero-kicker">Payment Verification OCR</span>
          <h1>Payment verification workflow for screenshot-driven transaction checks.</h1>
          <p>
            Upload registration sheets, map columns, launch OCR verification, and inspect the result
            set without the old Streamlit bottleneck.
          </p>
        </section>

        <StepRail currentStep={currentStep} />

        {errorMessage ? <div className="error-banner glass-panel">{errorMessage}</div> : null}

        {currentStep === "upload" ? (
          <section className="wizard-card glass-panel fade-in-up">
            <div className="card-header">
              <div>
                <span className="card-eyebrow">Step 01</span>
                <h2>Upload source files</h2>
              </div>
              <button className="secondary-button upload-reset-button" onClick={resetFlow} type="button">
                Reset
              </button>
            </div>

            <div className="upload-grid">
              <label className="file-drop glass-panel">
                <span>Registration CSV or XLSX</span>
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={(event) => setRegistrationFile(event.target.files?.[0] ?? null)}
                />
                <strong>{registrationFile?.name ?? "Choose file"}</strong>
              </label>

              <label className="file-drop glass-panel">
                <span>Transaction report files</span>
                <input
                  type="file"
                  accept=".csv,.xlsx,.pdf"
                  multiple
                  onChange={(event) => setReportFiles(Array.from(event.target.files ?? []))}
                />
                <strong>
                  {reportFiles.length > 0 ? `${reportFiles.length} files selected` : "Choose files"}
                </strong>
              </label>
            </div>

            <div className="preview-list">
              {registrationFile ? <div className="chip">Registration: {registrationFile.name}</div> : null}
              {reportFiles.map((file) => (
                <div className="chip" key={file.name}>
                  Report: {file.name}
                </div>
              ))}
            </div>

            <div className="actions">
              <button className="primary-button" type="button" onClick={moveToMapping}>
                Continue to mapping
              </button>
            </div>
          </section>
        ) : null}

        {currentStep === "mapping" ? (
          <section className="wizard-card glass-panel fade-in-up">
            <div className="card-header">
              <div>
                <span className="card-eyebrow">Step 02</span>
                <h2>Map the incoming data</h2>
              </div>
              <button className="secondary-button" onClick={() => setCurrentStep("upload")} type="button">
                Back
              </button>
            </div>

            <div className="mapping-grid">
              <div className="glass-panel inset-panel">
                <h3>Registration preview</h3>
                <select
                  value={config.regTransactionIdColumn ?? ""}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, regTransactionIdColumn: event.target.value }))
                  }
                >
                  {(registrationPreview?.columns ?? []).map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
                <label className="toggle-row">
                  <input
                    checked={config.useFallback}
                    onChange={(event) =>
                      setConfig((current) => ({ ...current, useFallback: event.target.checked }))
                    }
                    type="checkbox"
                  />
                  Use registration transaction ID when OCR fails
                </label>
                <PreviewTable preview={registrationPreview} />
              </div>

              <div className="glass-panel inset-panel">
                <h3>Report preview</h3>
                <select
                  value={config.rrnColumn ?? ""}
                  onChange={(event) => setConfig((current) => ({ ...current, rrnColumn: event.target.value }))}
                >
                  {(reportPreview?.columns ?? []).map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
                <select
                  value={config.amountColumn ?? ""}
                  onChange={(event) => setConfig((current) => ({ ...current, amountColumn: event.target.value }))}
                >
                  {(reportPreview?.columns ?? []).map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
                <PreviewTable preview={reportPreview} />
              </div>
            </div>

            <div className="actions">
              <button className="secondary-button" type="button" onClick={() => setCurrentStep("upload")}>
                Back
              </button>
              <button className="primary-button" type="button" onClick={moveToProcessing}>
                Launch verification
              </button>
            </div>
          </section>
        ) : null}

        {currentStep === "processing" ? (
          <section className="wizard-card glass-panel pulse-glow fade-in-up">
            <div className="card-header">
              <div>
                <span className="card-eyebrow">Step 03</span>
                <h2>Processing run</h2>
              </div>
              <span className="status-chip">{runStatus?.status ?? "queued"}</span>
            </div>

            <div className="progress-shell">
              <div className="progress-bar">
                <div className="progress-value" style={{ width: `${runStatus?.progress ?? 0}%` }} />
              </div>
              <strong>{runStatus?.progress ?? 0}%</strong>
            </div>

            <p className="processing-message">{runStatus?.message ?? "Preparing run..."}</p>

            {runStatus?.status === "failed" ? (
              <div className="actions">
                <button className="secondary-button" onClick={resetFlow} type="button">
                  Start over
                </button>
              </div>
            ) : null}

            <button className="primary-button" disabled={isSubmitting || runStatus?.status !== "completed"} type="button" onClick={() => setCurrentStep("results")}>
              Open results
            </button>
          </section>
        ) : null}

        {currentStep === "results" ? (
          <section className="wizard-card glass-panel fade-in-up">
            <div className="card-header">
              <div>
                <span className="card-eyebrow">Step 04</span>
                <h2>Verification results</h2>
              </div>
              <div className="actions-inline">
                {runStatus ? (
                  <a className="secondary-button" href={`/api/runs/${runStatus.runId}/download`}>
                    Download CSV
                  </a>
                ) : null}
                <button className="primary-button" onClick={resetFlow} type="button">
                  New run
                </button>
              </div>
            </div>

            {runStatus?.summary ? <SummaryCards summary={runStatus.summary} /> : null}

            <div className="filter-row">
              {verificationFilters.map((filter) => (
                <button
                  className={filter === resultsFilter ? "filter-chip filter-chip-active" : "filter-chip"}
                  key={filter}
                  onClick={() => setFilter(filter)}
                  type="button"
                >
                  {filter}
                </button>
              ))}
            </div>

            <ResultsTable rows={results?.rows ?? []} />

            <div className="pagination-row">
              <button
                className="secondary-button"
                disabled={resultsPage <= 1}
                onClick={() => setResultsPage((page) => Math.max(page - 1, 1))}
                type="button"
              >
                Previous
              </button>
              <span>
                Page {results?.page ?? resultsPage} of{" "}
                {results ? Math.max(1, Math.ceil(results.total / results.pageSize)) : 1}
              </span>
              <button
                className="secondary-button"
                disabled={!results || results.page * results.pageSize >= results.total}
                onClick={() => setResultsPage((page) => page + 1)}
                type="button"
              >
                Next
              </button>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}

function PreviewTable({ preview }: { preview: PreviewData | null }) {
  if (!preview || preview.columns.length === 0) {
    return <p className="empty-copy">No preview available.</p>;
  }

  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            {preview.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, index) => (
            <tr key={index}>
              {preview.columns.map((column) => (
                <td key={column}>{row[column]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsTable({ rows }: { rows: Record<string, string>[] }) {
  if (rows.length === 0) {
    return <p className="empty-copy">No rows match the current filter.</p>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{row[column]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
