import type { RunSummary } from "../lib/types";

const metricLabels: [keyof RunSummary, string][] = [
  ["totalRecords", "Total Records"],
  ["verified", "Verified"],
  ["notVerified", "Not Verified"],
  ["noIdExtracted", "No ID Extracted"],
  ["duplicates", "Duplicates"],
  ["registrationDuplicates", "Registration Duplicates"],
  ["amountMismatch", "Amount Mismatch"]
];

export function SummaryCards({ summary }: { summary: RunSummary }) {
  return (
    <div className="summary-grid">
      {metricLabels.map(([key, label]) => (
        <div className="glass-panel floating-card metric-card" key={key}>
          <span className="metric-label">{label}</span>
          <strong className="metric-value">{summary[key]}</strong>
        </div>
      ))}
    </div>
  );
}
