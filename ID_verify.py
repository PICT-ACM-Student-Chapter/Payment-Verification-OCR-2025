# This file verifies whether the ID extracted exists in given input transaction records
# It also verifies for duplicate transaction ID
# The input file is the PDF file to be sent by the teacher
import pandas as pd
from pathlib import Path
import os
import json
import glob
import re
import numpy as np

# File locations:
TRANSACTION_REPORT_PATH = "TransactionReport.pdf"
EXTRACTED_DATA_PATH = "processed_transactions.csv"  # or .xlsx
OUTPUT_PATH = "verified_transactions.csv"  # or .xlsx
VERIFIED_DB_PATH = "verified_ID.csv"  # or .xlsx
# ---


def load_column_config():
    """Load column configuration from JSON file

    Returns a dict with optional overrides. Sensible defaults are provided so the
    rest of the pipeline can operate without a config file.  Keys:
    - rrn_column: explicit RRN/UTR column name if available
    - amount_column: amount column name if available
    - details_column: column containing narrative/details if rrn_column is absent
    """
    defaults = {
        "rrn_column": "rrn",
        "amount_column": "amount",
        "details_column": "details",
    }
    try:
        if os.path.exists("column_config.json"):
            with open("column_config.json", "r") as f:
                config = json.load(f)
                defaults.update({k: v for k, v in config.items() if v})
    except Exception:
        # Fall back to defaults silently
        pass
    return defaults


# ------------------------ Helpers for column + regex parsing -----------------

DETAILS_CANDIDATES = [
    "details",
    "transaction details",
    "narration",
    "description",
    "message",
    "info",
]

AMOUNT_CANDIDATES = [
    "amount",
    "credit",
    "debit",
    "value",
    "txn amount",
    "transaction amount",
]


def _lower_col_map(columns):
    """Return mapping of lowercase->original column names."""
    return {str(c).strip().lower(): c for c in columns}


def _find_column(df: pd.DataFrame, candidates):
    """Case-insensitive column finder for a list of candidate names."""
    cmap = _lower_col_map(df.columns)
    for cand in candidates:
        key = cand.lower()
        if key in cmap:
            return cmap[key]
    return None


def _clean_amount(value):
    """Convert various amount representations (₹1,234.00, '1234', 1234.5) to Int32."""
    if pd.isna(value):
        return pd.NA
    # Direct numeric handling
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(value))
    try:
        s = str(value)
        # Extract first currency-like number if there's text
        m = re.search(r"(?:₹|INR|Rs\.?|Amount\s*:)?\s*([0-9][0-9,]*)(?:\.\d{1,2})?", s)
        if m:
            return int(m.group(1).replace(",", ""))
        # Fallback numeric parse
        return int(float(s.replace(",", "").replace("₹", "")))
    except Exception:
        return pd.NA


UTR_PATTERNS = [
    # Common forms: "UTR No. 526815046824" or "UTR: 526815046824"
    re.compile(r"UTR\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", re.IGNORECASE),
    # Sometimes present as "Ref No/UPI Ref"
    re.compile(r"(?:Ref|Reference)\s*(?:No\.?|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", re.IGNORECASE),
    # Or plain long numeric id
    re.compile(r"\b([0-9]{11,22})\b"),
]


def _extract_rrn_from_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    for pat in UTR_PATTERNS:
        m = pat.search(text)
        if m:
            token = m.group(1)
            # Keep digits only when token is mixed; most verification uses numeric UTRs
            digits = re.sub(r"\D", "", token)
            if len(digits) >= 8:
                try:
                    return int(digits)
                except Exception:
                    return pd.NA
    return pd.NA


def _parse_details_rows(df: pd.DataFrame, details_col: str, amount_col_hint: str | None = None):
    """Build a canonical DataFrame with columns [rrn, amount] from a DataFrame
    that contains a 'details' column possibly embedding UTR/amount.
    """
    details_series = df[details_col].astype(str)

    # Try resolving amount column from provided hint or common candidates
    amount_col = None
    if amount_col_hint and amount_col_hint in df.columns:
        amount_col = amount_col_hint
    if not amount_col:
        amount_col = _find_column(df, AMOUNT_CANDIDATES)

    out = pd.DataFrame()
    out["rrn"] = details_series.apply(_extract_rrn_from_text).astype("Int64")

    if amount_col:
        out["amount"] = df[amount_col].apply(_clean_amount).astype("Int32")
    else:
        # Attempt amount extraction from details text as a fallback
        out["amount"] = details_series.apply(_clean_amount).astype("Int32")

    out = out.dropna(subset=["rrn"])  # Require RRN for verification
    return out

def input_report():
    """Processes all transaction report files and combines them into a single dataframe"""
    all_dfs = []
    
    # Check for single files
    single_files = [
        ("TransactionReport.xlsx", "xlsx"), 
        ("TransactionReport.csv", "csv"),
        ("TransactionReport.pdf", "pdf")
    ]
    
    # Check for numbered files
    numbered_files = []
    for ext in ["xlsx", "csv", "pdf"]:
        numbered_files.extend([(f, ext) for f in glob.glob(f"TransactionReport_*.{ext}")])
    
    # Combine all found files
    found_files = [(f, ext) for f, ext in single_files if os.path.exists(f)]
    found_files.extend(numbered_files)
    
    if not found_files:
        raise FileNotFoundError("No transaction report files found (CSV, Excel, or PDF)")
    
    print(f"Processing {len(found_files)} transaction report file(s)...")
    
    for file_path, file_type in found_files:
        print(f"Processing: {file_path}")
        
        if file_type == "xlsx":
            df = process_excel_report(file_path)
        elif file_type == "csv":
            df = process_csv_report(file_path)
        elif file_type == "pdf":
            df = process_pdf_report(file_path)
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No valid data found in transaction report files")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates based on RRN
    combined_df = combined_df.drop_duplicates(subset=['rrn'])
    
    print(f"Total unique transactions loaded: {len(combined_df)}")
    return combined_df

def process_excel_report(excel_path):
    """Processes Excel transaction report with custom column names or details parsing."""
    df = pd.read_excel(excel_path)
    cfg = load_column_config()

    # First try explicit columns if present
    rrn_col = _find_column(df, [cfg["rrn_column"]]) or _find_column(df, ["rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [cfg["amount_column"]]) or _find_column(df, AMOUNT_CANDIDATES)

    if rrn_col and amount_col and rrn_col in df.columns and amount_col in df.columns:
        df_processed = df[[rrn_col, amount_col]].copy()
        df_processed.columns = ["rrn", "amount"]
        df_processed["rrn"] = pd.to_numeric(df_processed["rrn"], errors="coerce").astype("Int64")
        df_processed["amount"] = df_processed["amount"].apply(_clean_amount).astype("Int32")
        return df_processed.dropna(subset=["rrn"])  # ensure rrn exists

    # Else parse from a details/narration column
    details_col = _find_column(df, [cfg["details_column"]] + DETAILS_CANDIDATES)
    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    raise ValueError(
        f"Could not find RRN/UTR columns or a Details/Narration column in {excel_path}. Available columns: {list(df.columns)}"
    )

def process_csv_report(csv_path):
    """Processes CSV transaction report with custom column names or details parsing."""
    df = pd.read_csv(csv_path)
    cfg = load_column_config()

    rrn_col = _find_column(df, [cfg["rrn_column"]]) or _find_column(df, ["rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [cfg["amount_column"]]) or _find_column(df, AMOUNT_CANDIDATES)

    if rrn_col and amount_col and rrn_col in df.columns and amount_col in df.columns:
        df_processed = df[[rrn_col, amount_col]].copy()
        df_processed.columns = ["rrn", "amount"]
        df_processed["rrn"] = pd.to_numeric(df_processed["rrn"], errors="coerce").astype("Int64")
        df_processed["amount"] = df_processed["amount"].apply(_clean_amount).astype("Int32")
        return df_processed.dropna(subset=["rrn"])  # ensure rrn exists

    details_col = _find_column(df, [cfg["details_column"]] + DETAILS_CANDIDATES)
    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    raise ValueError(
        f"Could not find RRN/UTR columns or a Details/Narration column in {csv_path}. Available columns: {list(df.columns)}"
    )


def process_pdf_report(pdf_path):
    """Processes PDF transaction report.

    Supports two shapes:
    1) Tabular PDFs with explicit RRN/Amount columns
    2) Tabular PDFs with a single Details/Narration column containing UTR and an
       Amount/Credit column elsewhere
    Falls back to text extraction if tables are not present.
    """
    try:
        import pdfplumber
    except ImportError:
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PDF processing requires either pdfplumber or PyPDF2. Install with: pip install pdfplumber")
    
    # Try pdfplumber first (better table extraction)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            all_rows = []
            header = None
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    if not header:
                        header = table[0]
                    # Some PhonePe PDFs repeat header per page; skip duplicates
                    data_rows = [row for row in table[1:] if any(cell is not None for cell in row)]
                    all_rows.extend(data_rows)
            if not all_rows or not header:
                raise ValueError(f"No table data found in PDF: {pdf_path}")
            # Convert to DataFrame
            df = pd.DataFrame(all_rows, columns=header)
            
    except (ImportError, Exception):
        # Fallback to PyPDF2 for text extraction
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # When only raw text is available, try to split the text into logical blocks
        # and extract UTR/amount pairs. We consider each occurrence of "UTR" as a new row.
        blocks = re.split(r"(?i)\b(?:UTR\s*(?:No\.?|Number|#)?\s*[:\-]?\s*|Transaction ID\s*)", text)
        # The split drops the marker; rebuild by scanning for matches
        utrs = re.findall(r"(?i)UTR\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", text)
        amts = re.findall(r"(?:₹|INR|Rs\.?)[\s]*([0-9][0-9,]*)", text)
        rows = []
        for i, u in enumerate(utrs):
            rrn = _extract_rrn_from_text(u)
            amt = _clean_amount(amts[i]) if i < len(amts) else pd.NA
            rows.append({"rrn": rrn, "amount": amt})
        df = pd.DataFrame(rows)
        if df.empty:
            raise NotImplementedError("PDF text parsing did not find UTR numbers; please provide a sample file to tune parsing.")

    # At this point we have a DataFrame "df" from pdfplumber tables.
    cfg = load_column_config()
    rrn_col = _find_column(df, [cfg["rrn_column"], "rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [cfg["amount_column"]] + AMOUNT_CANDIDATES)
    details_col = _find_column(df, [cfg["details_column"]] + DETAILS_CANDIDATES)

    if rrn_col and amount_col and rrn_col in df.columns and amount_col in df.columns:
        out = df[[rrn_col, amount_col]].copy()
        out.columns = ["rrn", "amount"]
        out["rrn"] = pd.to_numeric(out["rrn"], errors='coerce').astype("Int64")
        out["amount"] = out["amount"].apply(_clean_amount).astype("Int32")
        return out.dropna(subset=["rrn"])  # Ensure RRN exists

    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    # As a last resort, try to derive from any text-like column
    text_like_cols = [c for c in df.columns if df[c].astype(str).str.contains("UTR", case=False, na=False).any()]
    if text_like_cols:
        return _parse_details_rows(df, text_like_cols[0], amount_col_hint=amount_col)

    raise ValueError(f"Could not identify RRN/UTR or Details columns in PDF table. Available columns: {list(df.columns)}")


def id_verification(input_df, report_df):
    """Verifies the extracted transaction IDs with the ones in report"""
    # Adding verified/not verified
    input_df["Verification"] = input_df["extracted_transaction_id"].apply(
        lambda rrn: (
            "Verified"
            if rrn in set(report_df.dropna()["rrn"].astype(int))
            else "Not Verified"
        )
    )

    # Adding ID not found
    input_df.loc[input_df["extracted_transaction_id"].isna(), "Verification"] = (
        "No ID extracted"
    )

    return input_df


def read_verified_file():
    """Reads and returns the dataframe of Verified ID (csv or xlsx)"""
    verified_file = Path(VERIFIED_DB_PATH)
    # Check if file exists. Make new if not
    if verified_file.exists():
        if verified_file.suffix == ".xlsx":
            verified_df = pd.read_excel(verified_file, dtype={"rrn": int})
        else:
            verified_df = pd.read_csv(verified_file, dtype={"rrn": int})
    else:
        verified_df = pd.DataFrame(columns=["rrn"])
    return verified_df


def duplicate_check(input_df):
    """Checks for duplicate extracted IDs in verified IDs"""
    # Checking for duplicates
    verified_dataframe = read_verified_file()

    # Identifying duplicates
    input_df["duplicate"] = input_df["extracted_transaction_id"].apply(
        lambda rrn: rrn in set(verified_dataframe["rrn"].astype(int)) if pd.notna(rrn) else False
    )

    # Changing the "Verified" status of duplicates
    input_df.loc[input_df["duplicate"], "Verification"] = "Duplicate"
    
    # Appending the non duplicate verified entries to the verified database
    append_verified(input_df, verified_dataframe)
    
    input_df.drop(columns=["duplicate"], inplace=True)

    return input_df


def mismatch_check(input_df, report_df):
    """Checks for mismatch amount from the report"""
    
    # Skip amount mismatch check if 'amount' column doesn't exist in input_df
    # This is normal since extracted data from screenshots typically doesn't include amounts
    if 'amount' not in input_df.columns:
        print("Skipping amount mismatch check - no amount data in extracted transactions")
        return input_df
    
    # Create a lookup dictionary for report amounts for efficiency
    report_amounts = dict(zip(report_df['rrn'], report_df['amount']))
    
    def check_amount_mismatch(row):
        transaction_id = row['extracted_transaction_id']
        if pd.isna(transaction_id) or transaction_id not in report_amounts:
            return False
        
        input_amount = row.get('amount', None)
        if pd.isna(input_amount):
            return False
            
        report_amount = report_amounts[transaction_id]
        return input_amount != report_amount
    
    # Apply the mismatch check
    input_df["amtVerify"] = input_df.apply(check_amount_mismatch, axis=1)
    input_df.loc[input_df["amtVerify"], "Verification"] = "Amount mismatch"
    
    input_df.drop(columns=["amtVerify"], inplace=True)
    return input_df


def append_verified(input_df, verified_df):
    """Appends the unique verified IDs in input_df to verified_df and saves verified_df (csv or xlsx)"""
    # Get verified transactions that are NOT duplicates
    newly_verified_IDs = input_df.loc[
        (input_df["Verification"] == "Verified") & (~input_df["duplicate"]), 
        "extracted_transaction_id"
    ]
    
    if not newly_verified_IDs.empty:
        verified_df = pd.DataFrame(
            {
                "rrn": pd.concat(
                    [verified_df["rrn"], newly_verified_IDs],
                    ignore_index=True,
                ).drop_duplicates()
            }
        )
        if Path(VERIFIED_DB_PATH).suffix == ".xlsx":
            verified_df.to_excel(VERIFIED_DB_PATH, index=False)
        else:
            verified_df.to_csv(VERIFIED_DB_PATH, index=False)


def save(output_df: pd.DataFrame):
    if Path(OUTPUT_PATH).suffix == ".xlsx":
        output_df.to_excel(OUTPUT_PATH, index=False)
    else:
        output_df.to_csv(OUTPUT_PATH, index=False)


def main():
    # Input
    report_input = input_report()
    
    # Determine which extracted data file exists
    extracted_data_path = None
    if os.path.exists("processed_transactions.xlsx"):
        extracted_data_path = "processed_transactions.xlsx"
    elif os.path.exists("processed_transactions.csv"):
        extracted_data_path = "processed_transactions.csv"
    else:
        raise FileNotFoundError("No processed transactions file found (processed_transactions.csv or processed_transactions.xlsx)")
    
    # Read the extracted data file
    if extracted_data_path.endswith('.xlsx'):
        extracted_input = pd.read_excel(
            extracted_data_path,
            dtype={"extracted_transaction_id": "Int64"},
        )
    else:
        extracted_input = pd.read_csv(
            extracted_data_path,
            dtype={"extracted_transaction_id": "Int64"},
        )
    
    # Add amount column with proper dtype if it exists, otherwise skip amount verification
    if 'amount' in extracted_input.columns:
        extracted_input['amount'] = extracted_input['amount'].astype("Int32")
    else:
        print("No amount column found in extracted data - amount verification will be skipped")

    # Process
    # Order: ID verification > duplicates (append non duplicates) > amount mismatch
    # Example: if a transaction is marked duplicate it would not be checked for amount mismatch etc.
    df = id_verification(extracted_input, report_input)
    df = duplicate_check(df)
    df = mismatch_check(df, report_input)

    # Output
    save(df)


if __name__ == "__main__":
    main()
