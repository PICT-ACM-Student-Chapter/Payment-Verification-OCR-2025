import gc
import json
import os
import re
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import pytesseract
import requests

warnings.filterwarnings("ignore")

DEFAULT_MODEL_PATH = "model.pt"
DEFAULT_INPUT_PATH = "input.csv"
DEFAULT_OUTPUT_PATH = "processed_transactions.csv"

model = None
loaded_model_path = None
use_yolo = False


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


def load_yolo_model(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    global model, loaded_model_path, use_yolo

    if not os.path.exists(model_path):
        print("No YOLO model file found. Using fallback OCR method.")
        use_yolo = False
        return False

    if model is not None and loaded_model_path == model_path and use_yolo:
        return True

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        loaded_model_path = model_path
        use_yolo = True
        print("YOLO model loaded successfully")
        return True
    except Exception as exc:
        error_msg = str(exc)
        print(f"[x] YOLO model loading failed: {error_msg}")
        model = None
        loaded_model_path = None
        use_yolo = False
        return False


def find_id_box(img):
    global model, use_yolo

    if not use_yolo or model is None:
        return None

    try:
        results = model.predict(img, verbose=False)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        return boxes
    except Exception as exc:
        error_msg = str(exc)
        print(f"YOLO prediction failed: {error_msg}")
        if "'AAttn' object has no attribute 'qkv'" in error_msg:
            use_yolo = False
            model = None
        return None


def crop_image(img):
    try:
        boxes = find_id_box(img)
        if boxes is None:
            return img

        box = boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        height, width = img.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return img

        return img[y1:y2, x1:x2]
    except Exception as exc:
        print(f"Error in crop_image: {exc}")
        return img


def download_image(image_url: str):
    response = None
    try:
        response = requests.get(image_url, stream=True, timeout=20)
        if response.status_code != 200:
            return None

        image_array = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    finally:
        if response is not None:
            response.close()


def extract_transaction_details(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    candidates = []

    for line in lines:
        candidates.extend(re.findall(r"\b[A-Z0-9]{8,25}\b", line, flags=re.IGNORECASE))
        candidates.extend(re.findall(r"\b\d{12}\b", line))

    for candidate in candidates:
        cleaned = clean_transaction_id(candidate)
        if cleaned:
            return cleaned

    return None


def process_image_url(image_url: str):
    image = None
    cropped = None

    try:
        if not image_url:
            return None

        image = download_image(image_url)
        if image is None:
            return None

        cropped = crop_image(image)
        if cropped is None:
            return None

        text = pytesseract.image_to_string(cropped)
        extracted = extract_transaction_details(text)
        if extracted is not None:
            print(f"[EXTRACTED] url={image_url} id={extracted}")
        else:
            print(f"[NO_MATCH] url={image_url}")
        return extracted
    except Exception as exc:
        print(f"[ERROR] Failed to process image URL: {image_url}\nException: {exc}")
        return None
    finally:
        del image
        del cropped
        gc.collect()


def clean_transaction_id(transaction_id):
    if pd.isna(transaction_id) or not transaction_id:
        return None

    transaction_id = str(transaction_id).strip()
    transaction_id = re.sub(
        r"^(UTR|TXN|REF|ID)\s*[:\-#]?\s*",
        "",
        transaction_id,
        flags=re.IGNORECASE,
    )
    clean_id = re.sub(r"[^A-Za-z0-9]", "", transaction_id)

    if len(clean_id) < 8 or len(clean_id) > 25:
        return None

    if re.match(r"^\d{12}$", clean_id):
        return clean_id

    if len(clean_id) >= 12:
        return clean_id

    return None


def detect_screenshot_column(registration_df: pd.DataFrame) -> str:
    normalized = {str(column).strip().lower(): column for column in registration_df.columns}
    for candidate in ["screenshot", "screenshots", "screenshot_url", "image_url"]:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        "No screenshot column found. Expected one of: screenshot, screenshots, screenshot_url, image_url"
    )


def process_transactions(
    reg_path: str,
    config_path: str | None = None,
    column_config: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
):
    config = load_column_config(config_path=config_path, column_config=column_config)

    if reg_path.endswith(".xlsx"):
        reg = pd.read_excel(reg_path, dtype=str)
    else:
        reg = pd.read_csv(reg_path, dtype=str)

    screenshot_column = detect_screenshot_column(reg)
    reg_transaction_id_column = config.get("reg_transaction_id_column", "transactionId")
    use_fallback = config.get("use_fallback", True)

    if reg_transaction_id_column not in reg.columns:
        available_columns = {str(col).lower(): col for col in reg.columns}
        alt_column = available_columns.get(str(reg_transaction_id_column).lower())
        if alt_column:
            reg_transaction_id_column = alt_column
        else:
            use_fallback = False

    extracted_ids = []
    urls = reg[screenshot_column].fillna("").tolist()
    reg_transaction_ids = []

    if use_fallback and reg_transaction_id_column in reg.columns:
        reg_transaction_ids = reg[reg_transaction_id_column].fillna("").tolist()

    total = len(urls)
    for index, url in enumerate(urls, start=1):
        extracted_id = process_image_url(url)
        if extracted_id is None and use_fallback and index <= len(reg_transaction_ids):
            extracted_id = clean_transaction_id(reg_transaction_ids[index - 1])
        extracted_ids.append(extracted_id)
        if progress_callback is not None:
            progress_callback(index, total)

    reg["extracted_transaction_id"] = extracted_ids
    return reg


def save(df: pd.DataFrame, output_filename: str = DEFAULT_OUTPUT_PATH):
    df = df.copy()
    df["extracted_transaction_id"] = df["extracted_transaction_id"].astype(str)
    df["extracted_transaction_id"] = df["extracted_transaction_id"].replace("nan", None)

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_filename.endswith(".xlsx"):
        df.to_excel(output_filename, index=False)
    else:
        df.to_csv(output_filename, index=False)


def run_extraction(
    reg_path: str,
    output_path: str,
    config_path: str | None = None,
    column_config: dict | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    progress_callback: Callable[[int, int], None] | None = None,
):
    load_yolo_model(model_path)
    processed_df = process_transactions(
        reg_path,
        config_path=config_path,
        column_config=column_config,
        progress_callback=progress_callback,
    )
    save(processed_df, output_path)
    return processed_df


def main():
    input_path = None
    if os.path.exists("input.xlsx"):
        input_path = "input.xlsx"
    elif os.path.exists(DEFAULT_INPUT_PATH):
        input_path = DEFAULT_INPUT_PATH
    else:
        raise FileNotFoundError("No input file found (input.csv or input.xlsx)")

    run_extraction(
        reg_path=input_path,
        output_path=DEFAULT_OUTPUT_PATH,
        config_path="column_config.json",
        model_path=DEFAULT_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
