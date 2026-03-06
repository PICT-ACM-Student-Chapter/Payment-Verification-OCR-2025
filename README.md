# Payment-Verification-OCR-2025

Automated payment verification system using **OCR** and **AI-powered object detection** to extract and validate transaction IDs from payment screenshots.

## Features

- Extracts transaction IDs from payment screenshots using **YOLOv12** for cropping and **pytesseract** for OCR.
- Supports popular platforms: **PhonePe**, **Google Pay**, **Paytm**, **Amazon Pay**.
- Accepts transaction reports in **CSV**, **Excel**, and **PDF** formats.
- Automatically matches extracted transaction IDs with backend transaction logs.
- Detects **duplicate** and **registration duplicate** transaction IDs.
- Marks registrations as "Verified", "Not Verified", "Duplicate", "Registration Duplicate", "No ID extracted", or "Amount mismatch".
- **Streamlit Web UI** for easy file upload, column configuration, and result download.
- **Fallback mechanism**: Uses registration form transaction IDs when OCR fails.
- Docker support for deployment.

## Tech Stack

- **Python** — Core language
- **YOLOv12** — Object detection for cropping transaction ID regions
- **pytesseract** — OCR text extraction
- **Streamlit** — Web UI
- **Pandas** — Data processing
- **OpenCV** — Image processing
- **pdfplumber / PyPDF2** — PDF transaction report parsing
- **Docker** — Containerized deployment

## Project Structure

```
Payment-Verification-OCR-2025/
├── app.py                 # Streamlit web UI (main entry point)
├── extraction.py          # OCR extraction module (YOLO + pytesseract)
├── ID_verify.py           # Transaction ID verification module
├── pipeline.py            # CLI runner for batch processing
├── model.pt               # YOLOv12 model weights
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## How It Works

1. **Upload** the registration CSV/Excel file (with screenshot URLs) via the web UI.
2. **Upload** one or more transaction report files (CSV, Excel, or PDF).
3. **Configure** which columns contain transaction IDs, RRNs, and amounts.
4. The system:
   - Downloads each screenshot from the URLs.
   - Uses YOLOv12 to detect and crop the transaction ID region.
   - Applies pytesseract OCR to extract the transaction ID.
   - Falls back to the registration form ID if OCR fails.
   - Matches extracted IDs against the uploaded transaction reports.
   - Checks for duplicates and amount mismatches.
5. **Download** the final verification report as CSV.

## Quick Start

### Web UI (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### CLI (Batch Processing)

```bash
# Place input.csv (with "screenshot" column) in the project root
python pipeline.py
```


## Supported Receipt Types

| Platform     | ID Format                                      |
|-------------|------------------------------------------------|
| **PhonePe**  | UTR numbers starting with `T` + 21 digits      |
| **Google Pay**| Transaction IDs like `AXIS1234567890`          |
| **Paytm**    | 12–15 digit numeric reference numbers          |
| **Amazon Pay**| Bank Reference ID (alphanumeric)               |

## System Requirements

- Python 3.11+
- Tesseract OCR installed on the system
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
