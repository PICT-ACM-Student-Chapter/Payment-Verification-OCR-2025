import gc
import json
import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import pytesseract
import requests

# Suppress warnings that might cause issues
warnings.filterwarnings('ignore')

# --- IMP: Set the paths of the following properly
# Include the code to choose yolo model here
MODEL_PATH = "model.pt"
INPUT_PATH = "input.csv"  # or .xlsx
# Input format:
# column : "screenshots" with all screenshot URLs
OUTPUT_PATH = "processed_transactions.csv"  # or .xlsx
# ---

# Global model variable
model = None
use_yolo = False

def load_column_config():
    """Load column configuration from JSON file if it exists"""
    try:
        if os.path.exists("column_config.json"):
            with open("column_config.json", "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load column config: {e}")
    return {}

def load_yolo_model():
    """
    Safely loads the YOLO model with comprehensive error handling.
    Returns False if YOLO cannot be used, True if successful.
    """
    global model, use_yolo
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print("No YOLO model file found. Using fallback OCR method.")
        use_yolo = False
        return False
    
    try:
        # Import ultralytics only when needed
        from ultralytics import YOLO
        
        # Try to load the model lazily and cache in memory
        model = YOLO(MODEL_PATH)
        
        print("✅ YOLO model loaded successfully")
        use_yolo = True
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"[x] YOLO model loading failed: {error_msg}")
        
        # Check for specific compatibility errors
        if "'AAttn' object has no attribute 'qkv'" in error_msg:
            print("[i] This is a known compatibility issue with ultralytics versions.")
            print("   Using fallback OCR method without YOLO cropping.")
        elif "CUDA" in error_msg or "GPU" in error_msg:
            print("[i] GPU/CUDA issue detected. Using fallback OCR method.")
        else:
            print("[i] Unknown YOLO error. Using fallback OCR method.")
        
        model = None
        use_yolo = False
        return False

def find_id_box(img):
    """
    Runs the YOLO model on the input image and returns detected boxes or None.
    Includes comprehensive error handling for the 'AAttn' object error.
    """
    global use_yolo, model
    
    if not use_yolo or model is None:
        return None
    
    try:
        # Run prediction with error handling
        results = model.predict(img, verbose=False)
        
        if not results or len(results) == 0:
            return None
            
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
            
        return boxes
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ YOLO prediction failed: {error_msg}")
        
        # If we get the 'AAttn' error during prediction, disable YOLO for future calls
        if "'AAttn' object has no attribute 'qkv'" in error_msg:
            print("⚠️ Disabling YOLO due to compatibility issues. Using fallback OCR.")
            use_yolo = False
            model = None
        
        return None

def crop_image(img):
    """
    Crops the input image to the first detected YOLO bounding box.
    Falls back to full image if no box is detected or YOLO fails.
    """
    try:
        boxes = find_id_box(img)
        if boxes is None:
            print("No YOLO boxes detected, using full image for OCR")
            return img

        # Use the first detected box
        box = boxes.xyxy[0]  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box)

        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure valid crop dimensions
        if x2 <= x1 or y2 <= y1:
            print("Invalid crop dimensions, using full image")
            return img

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img
        
    except Exception as e:
        print(f"Error in crop_image: {e}")
        return img


def download_image(image_url):
    """
    Downloads an image from a given URL and returns it as a NumPy array.

    Args:
        image_url (str): URL of the image to download.

    Returns:
        img (np.ndarray or None): Decoded image as a NumPy array, or None if download fails.
    """
    response = None
    try:
        response = requests.get(image_url, stream=True, timeout=20)
        if response.status_code == 200:
            # Use frombuffer to avoid an extra copy
            buf = response.content
            image_array = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return img
        return None
    finally:
        if response is not None:
            response.close()


def process_image_url(image_url):
    """
    Processes an image URL: downloads, crops, runs OCR, and extracts the transaction ID.

    Args:
        image_url (str): URL of the image to process.

    Returns:
        transaction_id (str or None): Extracted transaction ID, or None if extraction fails.
    """
    image = None
    cropped = None
    try:
        if not image_url:
            return None
        image = download_image(image_url)
        if image is not None:
            cropped = crop_image(image)
            if cropped is None:
                return None
            text = pytesseract.image_to_string(cropped)
            extract = extract_transaction_details(text)
            if extract is not None:
                print(f"[EXTRACTED] url={image_url} id={extract}")
            else:
                print(f"[NO_MATCH] url={image_url}")
            return extract
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process image URL: {image_url}\nException: {e}")
        return None
    finally:
        # Explicitly free large arrays
        del image
        del cropped
        gc.collect()


def clean_transaction_id(transaction_id):
    """
    Clean and validate transaction ID from user input.
    Handles various formats like UTR numbers, transaction IDs, etc.
    
    Args:
        transaction_id: Raw transaction ID from user input
    
    Returns:
        str or None: Cleaned transaction ID if valid, None otherwise
    """
    if pd.isna(transaction_id) or not transaction_id:
        return None
    
    # Convert to string and clean
    transaction_id = str(transaction_id).strip()
    
    # Remove common prefixes and whitespace
    transaction_id = re.sub(r'^(UTR|TXN|REF|ID)\s*[:\-#]?\s*', '', transaction_id, flags=re.IGNORECASE)
    
    # Extract only alphanumeric characters
    clean_id = re.sub(r'[^A-Za-z0-9]', '', transaction_id)
    
    # Validate length (transaction IDs are typically 8-25 characters)
    if len(clean_id) >= 8 and len(clean_id) <= 25:
        # Check if it's purely numeric and meets 12-digit transaction ID pattern
        if re.match(r'^\d{12}$', clean_id):
            return clean_id
        # Or if it's a longer alphanumeric ID (like PhonePe T-series)
        elif len(clean_id) >= 12:
            return clean_id
    
    return None


def extract_transaction_details(text):
    """
    Extracts a 12-digit transaction ID from OCR text.

    Args:
        text (str): OCR-extracted text from the transaction screenshot.

    Returns:
        transaction_id (str or None): 12-digit transaction ID if found, else None.
    """
    lines = text.split("\n")
    transaction_id = None
    print(lines)
    if len(lines) == 2:
        transaction_id = lines[0][-12:]
    elif len(lines) == 3:
        transaction_id = lines[1]
    pattern = r"^\d{12}$"
    if re.match(pattern, transaction_id):
        return transaction_id
    else:
        return None


def process_transactions(reg_path):
    """
    Processes an input CSV/Excel file to extract transaction IDs from screenshots.
    If OCR/YOLO fails, uses fallback transaction ID from registration data.

    Args:
        reg_path (str): Path to the input CSV or Excel file.

    Returns:
        reg (pd.DataFrame): DataFrame with an added 'extracted_transaction_id' column.
    """
    # Load column configuration
    config = load_column_config()
    
    # Check file extension and read accordingly
    if reg_path.endswith('.xlsx'):
        reg = pd.read_excel(reg_path, dtype=str)
    else:
        reg = pd.read_csv(reg_path, dtype=str)
    
    # Get column names from config
    reg_transaction_id_column = config.get('reg_transaction_id_column', 'transactionId')
    use_fallback = config.get('use_fallback', True)
    
    # Check if the specified column exists
    if reg_transaction_id_column not in reg.columns:
        # Try to find a similar column (case insensitive)
        available_columns = {col.lower(): col for col in reg.columns}
        alt_column = available_columns.get(reg_transaction_id_column.lower())
        if alt_column:
            reg_transaction_id_column = alt_column
        else:
            print(f"Warning: Column '{reg_transaction_id_column}' not found. Available columns: {list(reg.columns)}")
            print("Fallback mechanism will not be used.")
            use_fallback = False
    
    # Process row by row to limit memory; do not retain image arrays
    extracted_ids = []
    urls = reg["screenshot"].fillna("").tolist()
    
    # Get registration transaction IDs for fallback
    reg_transaction_ids = []
    if use_fallback and reg_transaction_id_column in reg.columns:
        reg_transaction_ids = reg[reg_transaction_id_column].fillna("").tolist()
        print(f"Using fallback transaction IDs from column: {reg_transaction_id_column}")
    
    total = len(urls)
    
    for idx, url in enumerate(urls, start=1):
        print(f"[PROCESS] {idx}/{total} url={url}")
        
        # Try OCR/YOLO extraction first
        extracted_id = process_image_url(url)
        
        # If OCR failed and fallback is enabled, use registration transaction ID
        if extracted_id is None and use_fallback and idx <= len(reg_transaction_ids):
            fallback_id = clean_transaction_id(reg_transaction_ids[idx-1])
            if fallback_id:
                print(f"[FALLBACK] OCR failed, using registration ID: {fallback_id}")
                extracted_id = fallback_id
            else:
                print(f"[FALLBACK] Invalid registration ID: {reg_transaction_ids[idx-1]}")
        
        extracted_ids.append(extracted_id)
    
    reg["extracted_transaction_id"] = extracted_ids
    return reg


def save(df, output_filename="processed_transactions.csv"):
    """
    Saves the processed DataFrame to a CSV or Excel file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_filename (str): Output filename (CSV or Excel).
    """
    # Keep transaction IDs as strings to support alphanumeric IDs like MPIB3332191571
    df["extracted_transaction_id"] = df["extracted_transaction_id"].astype(str)
    df["extracted_transaction_id"] = df["extracted_transaction_id"].replace('nan', None)
    
    # Save based on file extension
    if output_filename.endswith('.xlsx'):
        df.to_excel(output_filename, index=False)
    else:
        df.to_csv(output_filename, index=False)


def main():
    """
    Main function to handle input, processing, and output.
    """
    # Initialize YOLO model with error handling
    print("🔍 Initializing YOLO model...")
    yolo_success = load_yolo_model()
    
    if yolo_success:
        print("✅ Using YOLO model for transaction ID detection")
    else:
        print("📝 Using fallback OCR method (full image processing)")
    
    # Determine input file path (check both .csv and .xlsx)
    input_path = None
    if os.path.exists("input.xlsx"):
        input_path = "input.xlsx"
    elif os.path.exists("input.csv"):
        input_path = "input.csv"
    else:
        raise FileNotFoundError("No input file found (input.csv or input.xlsx)")
    
    print(f"📁 Using input file: {input_path}")
    
    # Process transactions
    print("🔄 Processing transactions...")
    processed_df = process_transactions(input_path)
    save(processed_df, OUTPUT_PATH)
    print("✅ Transaction processing completed!")


if __name__ == "__main__":
    main()
