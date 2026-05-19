import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_PATH = "verified_transactions.csv"
DEFAULT_VERIFIED_DB_PATH = "verified_ID.csv"

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

UTR_PATTERNS = [
    re.compile(r"UTR\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", re.IGNORECASE),
    re.compile(r"(?:Ref|Reference)\s*(?:No\.?|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", re.IGNORECASE),
    re.compile(r"\b([0-9]{11,22})\b"),
]


def load_column_config(config_path: str | None = None, column_config: dict | None = None) -> dict:
    if column_config is not None:
        return column_config

    if config_path is None:
        config_path = "column_config.json"

    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
    except Exception as exc:
        print(f"Warning: Could not load column config: {exc}")

    return {}


def _lower_col_map(columns):
    return {str(column).strip().lower(): column for column in columns}


def _find_column(df: pd.DataFrame, candidates):
    col_map = _lower_col_map(df.columns)
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate).lower()
        if key in col_map:
            return col_map[key]
    return None


def _clean_amount(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(value))

    try:
        string_value = str(value)
        match = re.search(r"(?:₹|INR|Rs\.?|Amount\s*:)?\s*([0-9][0-9,]*)(?:\.\d{1,2})?", string_value)
        if match:
            return int(match.group(1).replace(",", ""))
        return int(float(string_value.replace(",", "").replace("₹", "")))
    except Exception:
        return pd.NA


def _extract_rrn_from_text_string(text: str):
    if not isinstance(text, str):
        text = str(text)

    for pattern in UTR_PATTERNS:
        match = pattern.search(text)
        if match:
            token = match.group(1).strip()
            if len(token) >= 8:
                return token
    return pd.NA


def _parse_details_rows(df: pd.DataFrame, details_col: str, amount_col_hint: str | None = None):
    details_series = df[details_col].astype(str)

    amount_col = None
    if amount_col_hint and amount_col_hint in df.columns:
        amount_col = amount_col_hint
    if not amount_col:
        amount_col = _find_column(df, AMOUNT_CANDIDATES)

    out = pd.DataFrame()
    out["rrn"] = details_series.apply(_extract_rrn_from_text_string)

    if amount_col:
        out["amount"] = pd.to_numeric(df[amount_col].apply(_clean_amount), errors="coerce").astype("Int64")
    else:
        out["amount"] = pd.to_numeric(details_series.apply(_clean_amount), errors="coerce").astype("Int64")

    return out.dropna(subset=["rrn"])


def check_registration_duplicates(input_df: pd.DataFrame, config: dict):
    reg_transaction_id_column = config.get("reg_transaction_id_column", "transactionId")

    if reg_transaction_id_column not in input_df.columns:
        return input_df

    def clean_reg_id(transaction_id):
        if pd.isna(transaction_id) or not transaction_id:
            return None
        transaction_id = str(transaction_id).strip()
        transaction_id = re.sub(r"^(UTR|TXN|REF|ID)\s*[:\-#]?\s*", "", transaction_id, flags=re.IGNORECASE)
        transaction_id = re.sub(r"[^A-Za-z0-9]", "", transaction_id)
        return transaction_id if len(transaction_id) >= 8 else None

    df_copy = input_df.copy()
    df_copy["_clean_reg_id"] = df_copy[reg_transaction_id_column].apply(clean_reg_id)
    reg_id_counts = Counter([rid for rid in df_copy["_clean_reg_id"].tolist() if rid is not None])
    duplicate_reg_ids = {rid for rid, count in reg_id_counts.items() if count > 1}

    if duplicate_reg_ids:
        duplicate_mask = df_copy["_clean_reg_id"].isin(duplicate_reg_ids)
        input_df.loc[duplicate_mask, "Verification"] = "Registration Duplicate"

    return input_df


def process_excel_report(excel_path: str, config: dict):
    df = pd.read_excel(excel_path)
    rrn_col = _find_column(df, [config.get("rrn_column", ""), "rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [config.get("amount_column", "")] + AMOUNT_CANDIDATES)

    if rrn_col and amount_col:
        out = df[[rrn_col, amount_col]].copy()
        out.columns = ["rrn", "amount"]
        out["rrn"] = out["rrn"].astype(str).str.strip()
        out["amount"] = out["amount"].apply(_clean_amount).astype("Int32")
        return out.dropna(subset=["rrn"])

    details_col = _find_column(df, [config.get("details_column", "")] + DETAILS_CANDIDATES)
    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    raise ValueError(f"Could not find RRN/UTR columns or a Details/Narration column in {excel_path}.")


def process_csv_report(csv_path: str, config: dict):
    df = pd.read_csv(csv_path)
    rrn_col = _find_column(df, [config.get("rrn_column", ""), "rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [config.get("amount_column", "")] + AMOUNT_CANDIDATES)

    if rrn_col and amount_col:
        out = df[[rrn_col, amount_col]].copy()
        out.columns = ["rrn", "amount"]
        out["rrn"] = out["rrn"].astype(str).str.strip()
        out["amount"] = out["amount"].apply(_clean_amount).astype("Int32")
        return out.dropna(subset=["rrn"])

    details_col = _find_column(df, [config.get("details_column", "")] + DETAILS_CANDIDATES)
    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    raise ValueError(f"Could not find RRN/UTR columns or a Details/Narration column in {csv_path}.")


def process_pdf_report(pdf_path: str, config: dict):
    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None

    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_rows = []
                header = None
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        if not header:
                            header = table[0]
                        all_rows.extend([row for row in table[1:] if any(cell is not None for cell in row)])
                if all_rows and header:
                    df = pd.DataFrame(all_rows, columns=header)
                else:
                    raise ValueError("No table data found")
        except Exception:
            df = None
    else:
        df = None

    if df is None:
        import PyPDF2

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)

        utrs = re.findall(r"(?i)UTR\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Z0-9]{8,22})", text)
        amounts = re.findall(r"(?:₹|INR|Rs\.?)[\s]*([0-9][0-9,]*)", text)
        rows = []
        for index, utr in enumerate(utrs):
            rows.append(
                {
                    "rrn": utr.strip(),
                    "amount": _clean_amount(amounts[index]) if index < len(amounts) else pd.NA,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("PDF text parsing did not find UTR numbers.")

    rrn_col = _find_column(df, [config.get("rrn_column", ""), "rrn", "utr", "utr no", "utr number"])
    amount_col = _find_column(df, [config.get("amount_column", "")] + AMOUNT_CANDIDATES)
    details_col = _find_column(df, [config.get("details_column", "")] + DETAILS_CANDIDATES)

    if rrn_col and amount_col:
        out = df[[rrn_col, amount_col]].copy()
        out.columns = ["rrn", "amount"]
        out["rrn"] = out["rrn"].astype(str).str.strip()
        out["amount"] = out["amount"].apply(_clean_amount).astype("Int32")
        return out.dropna(subset=["rrn"])

    if details_col:
        return _parse_details_rows(df, details_col, amount_col_hint=amount_col)

    raise ValueError(f"Could not identify RRN/UTR or Details columns in PDF table for {pdf_path}.")


def input_report(report_paths: list[str], config: dict):
    all_dfs = []

    for report_path in report_paths:
        suffix = Path(report_path).suffix.lower()
        if suffix == ".xlsx":
            df = process_excel_report(report_path, config)
        elif suffix == ".csv":
            df = process_csv_report(report_path, config)
        elif suffix == ".pdf":
            df = process_pdf_report(report_path, config)
        else:
            raise ValueError(f"Unsupported transaction report file type: {report_path}")

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid data found in transaction report files")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df["rrn"] = combined_df["rrn"].astype(str).str.strip()
    return combined_df.drop_duplicates(subset=["rrn"])


def read_verified_file(verified_db_path: str):
    verified_file = Path(verified_db_path)
    if verified_file.exists():
        if verified_file.suffix == ".xlsx":
            verified_df = pd.read_excel(verified_file, dtype={"rrn": str})
        else:
            verified_df = pd.read_csv(verified_file, dtype={"rrn": str})
        verified_df["rrn"] = verified_df["rrn"].astype(str).str.strip()
        return verified_df

    return pd.DataFrame(columns=["rrn"])


def append_verified(input_df: pd.DataFrame, verified_df: pd.DataFrame, verified_db_path: str):
    newly_verified_ids = input_df.loc[
        (input_df["Verification"] == "Verified") & (~input_df["duplicate"]),
        "extracted_transaction_id",
    ]

    if newly_verified_ids.empty:
        return

    verified_df = pd.DataFrame(
        {"rrn": pd.concat([verified_df["rrn"], newly_verified_ids], ignore_index=True).drop_duplicates()}
    )
    Path(verified_db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(verified_db_path).suffix == ".xlsx":
        verified_df.to_excel(verified_db_path, index=False)
    else:
        verified_df.to_csv(verified_db_path, index=False)


def id_verification(input_df: pd.DataFrame, report_df: pd.DataFrame):
    valid_rrns = set(report_df.dropna()["rrn"].astype(str).str.strip())
    input_df["Verification"] = input_df["extracted_transaction_id"].apply(
        lambda rrn: "Verified" if pd.notna(rrn) and str(rrn).strip() in valid_rrns else "Not Verified"
    )
    input_df.loc[input_df["extracted_transaction_id"].isna(), "Verification"] = "No ID extracted"
    return input_df


def duplicate_check(input_df: pd.DataFrame, verified_db_path: str):
    verified_dataframe = read_verified_file(verified_db_path)
    verified_rrns = set(verified_dataframe["rrn"].astype(str).str.strip()) if not verified_dataframe.empty else set()
    input_df["duplicate"] = input_df["extracted_transaction_id"].apply(
        lambda rrn: pd.notna(rrn) and str(rrn).strip() in verified_rrns
    )
    input_df.loc[input_df["duplicate"], "Verification"] = "Duplicate"
    append_verified(input_df, verified_dataframe, verified_db_path)
    input_df.drop(columns=["duplicate"], inplace=True)
    return input_df


def mismatch_check(input_df: pd.DataFrame, report_df: pd.DataFrame):
    if "amount" not in input_df.columns:
        return input_df

    report_amounts = dict(zip(report_df["rrn"].astype(str), report_df["amount"]))

    def check_amount_mismatch(row):
        transaction_id = row["extracted_transaction_id"]
        if pd.isna(transaction_id):
            return False
        transaction_id_str = str(transaction_id).strip()
        if transaction_id_str not in report_amounts:
            return False
        input_amount = row.get("amount", None)
        if pd.isna(input_amount):
            return False
        return input_amount != report_amounts[transaction_id_str]

    input_df["amtVerify"] = input_df.apply(check_amount_mismatch, axis=1)
    input_df.loc[input_df["amtVerify"], "Verification"] = "Amount mismatch"
    input_df.drop(columns=["amtVerify"], inplace=True)
    return input_df


def save(output_df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_PATH):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(output_path).suffix == ".xlsx":
        output_df.to_excel(output_path, index=False)
    else:
        output_df.to_csv(output_path, index=False)


def run_verification(
    extracted_data_path: str,
    report_paths: list[str],
    output_path: str,
    verified_db_path: str = DEFAULT_VERIFIED_DB_PATH,
    config_path: str | None = None,
    column_config: dict | None = None,
):
    config = load_column_config(config_path=config_path, column_config=column_config)
    report_input = input_report(report_paths, config)

    if extracted_data_path.endswith(".xlsx"):
        extracted_input = pd.read_excel(extracted_data_path, dtype={"extracted_transaction_id": str})
    else:
        extracted_input = pd.read_csv(extracted_data_path, dtype={"extracted_transaction_id": str})

    extracted_input["extracted_transaction_id"] = extracted_input["extracted_transaction_id"].replace("nan", None)
    extracted_input["extracted_transaction_id"] = extracted_input["extracted_transaction_id"].replace("None", None)

    if "amount" in extracted_input.columns:
        extracted_input["amount"] = extracted_input["amount"].astype("Int32")

    df = id_verification(extracted_input, report_input)
    df = check_registration_duplicates(df, config)
    df = duplicate_check(df, verified_db_path)
    df = mismatch_check(df, report_input)
    save(df, output_path)
    return df


def discover_default_report_paths():
    paths = []
    for filename in ["TransactionReport.xlsx", "TransactionReport.csv", "TransactionReport.pdf"]:
        if os.path.exists(filename):
            paths.append(filename)

    for extension in ["xlsx", "csv", "pdf"]:
        paths.extend(glob.glob(f"TransactionReport_*.{extension}"))

    if not paths:
        raise FileNotFoundError("No transaction report files found (CSV, Excel, or PDF)")

    return paths


def main():
    extracted_data_path = None
    if os.path.exists("processed_transactions.xlsx"):
        extracted_data_path = "processed_transactions.xlsx"
    elif os.path.exists("processed_transactions.csv"):
        extracted_data_path = "processed_transactions.csv"
    else:
        raise FileNotFoundError(
            "No processed transactions file found (processed_transactions.csv or processed_transactions.xlsx)"
        )

    run_verification(
        extracted_data_path=extracted_data_path,
        report_paths=discover_default_report_paths(),
        output_path=DEFAULT_OUTPUT_PATH,
        verified_db_path=DEFAULT_VERIFIED_DB_PATH,
        config_path="column_config.json",
    )


if __name__ == "__main__":
    main()
