# Imports
import cv2
import numpy as np
import re
import pytesseract
import requests
import pandas as pd
import os
import warnings
import traceback
import gc
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        print(f"❌ YOLO model loading failed: {error_msg}")
        
        # Check for specific compatibility errors
        if "'AAttn' object has no attribute 'qkv'" in error_msg:
            print("⚠️ This is a known compatibility issue with ultralytics versions.")
            print("   Using fallback OCR method without YOLO cropping.")
        elif "CUDA" in error_msg or "GPU" in error_msg:
            print("⚠️ GPU/CUDA issue detected. Using fallback OCR method.")
        else:
            print("⚠️ Unknown YOLO error. Using fallback OCR method.")
        
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
        logger.info(f"🔄 Attempting to download image from: {image_url}")
        
        # Validate URL
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"❌ Invalid URL format: {image_url}")
            return None
        
        # Attempt to download
        logger.debug(f"Making GET request with 20s timeout...")
        response = requests.get(image_url, stream=True, timeout=20)
        
        logger.info(f"📡 Response status code: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('content-type', 'unknown')
            logger.info(f"📄 Content type: {content_type}")
            
            if not content_type.startswith('image/'):
                logger.warning(f"⚠️ Unexpected content type: {content_type}")
            
            # Get content length
            content_length = response.headers.get('content-length')
            if content_length:
                logger.info(f"📊 Content length: {content_length} bytes")
            
            # Use frombuffer to avoid an extra copy
            buf = response.content
            logger.info(f"📥 Downloaded {len(buf)} bytes")
            
            if len(buf) == 0:
                logger.error(f"❌ Downloaded content is empty")
                return None
            
            # Decode image
            logger.debug("🖼️ Attempting to decode image...")
            image_array = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error(f"❌ Failed to decode image - cv2.imdecode returned None")
                logger.error(f"   Buffer size: {len(buf)} bytes")
                logger.error(f"   First 50 bytes: {buf[:50]}")
                return None
            
            height, width = img.shape[:2]
            logger.info(f"✅ Successfully decoded image: {width}x{height} pixels")
            return img
        else:
            logger.error(f"❌ HTTP request failed with status {response.status_code}")
            if hasattr(response, 'reason'):
                logger.error(f"   Reason: {response.reason}")
            if hasattr(response, 'text'):
                logger.error(f"   Response text: {response.text[:200]}...")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Timeout error downloading image from {image_url}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"🔗 Connection error downloading image from {image_url}: {str(e)}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"🌐 HTTP error downloading image from {image_url}: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"📡 Request error downloading image from {image_url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"💥 Unexpected error downloading image from {image_url}: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None
    finally:
        if response is not None:
            try:
                response.close()
                logger.debug("🔒 Response connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing response: {str(e)}")


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
        logger.info(f"🔍 Starting processing for URL: {image_url}")
        
        if not image_url:
            logger.warning("⚠️ Empty or None image URL provided")
            return None
            
        # Step 1: Download image
        logger.debug("📥 Step 1: Downloading image...")
        image = download_image(image_url)
        
        if image is None:
            logger.error(f"❌ Failed to download image from: {image_url}")
            return None
        
        logger.info(f"✅ Successfully downloaded image: {image.shape}")
        
        # Step 2: Crop image (YOLO detection or fallback)
        logger.debug("✂️ Step 2: Cropping image...")
        cropped = crop_image(image)
        
        if cropped is None:
            logger.error(f"❌ Failed to crop image from: {image_url}")
            return None
            
        logger.info(f"✅ Successfully cropped image: {cropped.shape}")
        
        # Step 3: OCR
        logger.debug("📖 Step 3: Running OCR...")
        text = pytesseract.image_to_string(cropped)
        logger.info(f"📝 OCR extracted text length: {len(text)} characters")
        logger.debug(f"📝 OCR text preview: {text[:100]}...")
        
        # Step 4: Extract transaction details
        logger.debug("🔍 Step 4: Extracting transaction details...")
        extract = extract_transaction_details(text)
        
        if extract is not None:
            logger.info(f"🎉 [EXTRACTED] url={image_url} id={extract}")
        else:
            logger.warning(f"❌ [NO_MATCH] url={image_url}")
            logger.debug(f"Full OCR text for analysis: {text}")
            
        return extract
        
    except Exception as e:
        logger.error(f"💥 Unexpected error processing {image_url}: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process image URL: {image_url}\nException: {e}")
        return None
    finally:
        # Explicitly free large arrays
        del image
        del cropped
        gc.collect()


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

    Args:
        reg_path (str): Path to the input CSV or Excel file.

    Returns:
        reg (pd.DataFrame): DataFrame with an added 'extracted_transaction_id' column.
    """
    try:
        logger.info(f"📂 Loading registration data from: {reg_path}")
        
        # Check file extension and read accordingly
        if reg_path.endswith('.xlsx'):
            reg = pd.read_excel(reg_path, dtype=str)
            logger.info("📊 Loaded Excel file successfully")
        else:
            reg = pd.read_csv(reg_path, dtype=str)
            logger.info("📊 Loaded CSV file successfully")
        
        logger.info(f"📈 Total records to process: {len(reg)}")
        
        # Check for screenshot column
        if "screenshot" not in reg.columns:
            available_cols = list(reg.columns)
            logger.error(f"❌ 'screenshot' column not found. Available columns: {available_cols}")
            raise ValueError(f"'screenshot' column not found in {reg_path}")
        
        # Process row by row to limit memory; do not retain image arrays
        extracted_ids = []
        urls = reg["screenshot"].fillna("").tolist()
        total = len(urls)
        
        # Statistics tracking
        successful_extractions = 0
        failed_downloads = 0
        failed_extractions = 0
        empty_urls = 0
        
        logger.info(f"🔄 Starting to process {total} image URLs...")
        
        for idx, url in enumerate(urls, start=1):
            logger.info(f"📍 [PROCESS] {idx}/{total} url={url}")
            
            if not url or url.strip() == "":
                logger.warning(f"⚠️ Empty URL at row {idx}")
                empty_urls += 1
                extracted_ids.append(None)
                continue
                
            try:
                extracted_id = process_image_url(url)
                extracted_ids.append(extracted_id)
                
                if extracted_id is not None:
                    successful_extractions += 1
                    logger.info(f"✅ Successfully extracted ID: {extracted_id}")
                else:
                    failed_extractions += 1
                    logger.warning(f"❌ Failed to extract ID from URL")
                    
            except Exception as e:
                logger.error(f"💥 Error processing URL {idx}: {str(e)}")
                failed_downloads += 1
                extracted_ids.append(None)
                
            # Progress update every 10 items
            if idx % 10 == 0:
                success_rate = (successful_extractions / idx) * 100
                logger.info(f"📊 Progress: {idx}/{total} ({success_rate:.1f}% success rate)")
        
        # Final statistics
        logger.info("📊 Final processing statistics:")
        logger.info(f"   ✅ Successful extractions: {successful_extractions}")
        logger.info(f"   ❌ Failed extractions: {failed_extractions}")
        logger.info(f"   📡 Failed downloads: {failed_downloads}")
        logger.info(f"   ⚠️ Empty URLs: {empty_urls}")
        
        success_rate = (successful_extractions / total) * 100 if total > 0 else 0
        logger.info(f"   📈 Overall success rate: {success_rate:.1f}%")
        
        reg["extracted_transaction_id"] = extracted_ids
        logger.info(f"✅ Successfully processed all {total} records")
        return reg
        
    except Exception as e:
        logger.error(f"💥 Fatal error in process_transactions: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise


def save(df, output_filename="processed_transactions.csv"):
    """
    Saves the processed DataFrame to a CSV or Excel file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_filename (str): Output filename (CSV or Excel).
    """
    df["extracted_transaction_id"] = df["extracted_transaction_id"].astype("Int64")
    
    # Save based on file extension
    if output_filename.endswith('.xlsx'):
        df.to_excel(output_filename, index=False)
    else:
        df.to_csv(output_filename, index=False)


def main():
    """
    Main function to handle input, processing, and output.
    """
    try:
        logger.info("🚀 Starting extraction process...")
        
        # Initialize YOLO model with error handling
        logger.info("🔍 Initializing YOLO model...")
        yolo_success = load_yolo_model()
        
        if yolo_success:
            logger.info("✅ Using YOLO model for transaction ID detection")
        else:
            logger.info("📝 Using fallback OCR method (full image processing)")
        
        # Determine input file path (check both .csv and .xlsx)
        input_path = None
        if os.path.exists("input.xlsx"):
            input_path = "input.xlsx"
            logger.info("📁 Found Excel input file: input.xlsx")
        elif os.path.exists("input.csv"):
            input_path = "input.csv"
            logger.info("📁 Found CSV input file: input.csv")
        else:
            error_msg = "No input file found (input.csv or input.xlsx)"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Process transactions
        logger.info("🔄 Starting transaction processing...")
        processed_df = process_transactions(input_path)
        
        if processed_df is not None and len(processed_df) > 0:
            save(processed_df, OUTPUT_PATH)
            logger.info(f"✅ Successfully processed {len(processed_df)} transactions and saved to {OUTPUT_PATH}")
        else:
            logger.error("❌ No transactions were processed successfully")
            
    except Exception as e:
        logger.error(f"💥 Fatal error in main process: {str(e)}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
