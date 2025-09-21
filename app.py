# app.py
import streamlit as st
import pandas as pd
import os
import glob
import extraction
import ID_verify
from datetime import datetime
import time
import numpy as np # Added for dummy image in YOLO model testing

# Page configuration
st.set_page_config(
    page_title="Payment Verification OCR",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    
    .status-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .file-uploader {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>💰 Payment Verification OCR System</h1>
    <p>Automated payment verification using AI-powered OCR and transaction matching</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.markdown("### 📋 Navigation")
    page = st.selectbox(
        "Choose a section:",
        ["🏠 Main Dashboard", "📊 Results", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 System Status")
    
    # Lightweight check only; defer model loading to first use
    model_exists = os.path.exists("model.pt")
    if model_exists:
        st.info("YOLO model present. It will load on first use.")
    else:
        st.info("ℹ️ No YOLO Model - Will use fallback OCR")
    
    st.markdown("---")
    st.markdown("### 🛠️ Supported Platforms")
    st.markdown("- **PhonePe**: UTR numbers")
    st.markdown("- **Google Pay**: Transaction IDs")
    st.markdown("- **Paytm**: Reference numbers")
    st.markdown("- **Amazon Pay**: Bank Reference IDs")

# Main dashboard
if page == "🏠 Main Dashboard":
    # File upload section
    st.markdown("## 📁 Upload Files")
    # Initialize processing guard
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Registration Data")
        uploaded_csv = st.file_uploader(
            "Upload User Registration CSV/Excel",
            type=["csv", "xlsx"],
            help="Upload the CSV or Excel file containing user registration data with screenshot URLs (Max 200MB per file)"
        )
        if uploaded_csv is not None:
            st.success(f"✅ Uploaded: {uploaded_csv.name}")
            # Show preview
            try:
                if uploaded_csv.name.endswith('.xlsx'):
                    df_preview = pd.read_excel(uploaded_csv)
                else:
                    df_preview = pd.read_csv(uploaded_csv)
                st.markdown("**Preview of uploaded data:**")
                st.dataframe(df_preview.head(), width="expand")
            except:
                st.warning("Could not preview file")

    with col2:
        st.markdown("### Transaction Reports")
        uploaded_files = st.file_uploader(
            "Upload Transaction Reports",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help="Upload CSV or Excel files containing transaction reports to verify against (Max 200MB per file)"
        )
        
        # Column selection for CSV/Excel files
        rrn_column = None
        amount_column = None
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
            
            # Show file list
            for i, file in enumerate(uploaded_files):
                file_type = ""
                st.write(f"{file_type} {file.name}")
            
            # Check if any CSV/Excel files are uploaded for column selection
            csv_excel_files = uploaded_files  # All files are now CSV/Excel
            
            if csv_excel_files:
                st.markdown("#### Column Configuration")
                st.info("📝 For CSV/Excel files, please specify which columns contain the RRN and Amount data:")
                
                # Preview the first CSV/Excel file to show available columns
                first_data_file = csv_excel_files[0]
                try:
                    if first_data_file.name.endswith('.xlsx'):
                        preview_df = pd.read_excel(first_data_file)
                    else:
                        preview_df = pd.read_csv(first_data_file)
                    
                    available_columns = list(preview_df.columns)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        rrn_column = st.selectbox(
                            "RRN Column:",
                            options=available_columns,
                            index=0 if 'rrn' not in [col.lower() for col in available_columns] else [col.lower() for col in available_columns].index('rrn'),
                            help="Select the column containing RRN/Transaction ID numbers"
                        )
                    
                    with col_b:
                        amount_column = st.selectbox(
                            "Amount Column:",
                            options=available_columns,
                            index=0 if 'amount' not in [col.lower() for col in available_columns] else [col.lower() for col in available_columns].index('amount'),
                            help="Select the column containing transaction amounts"
                        )
                    
                    # Show preview with selected columns
                    st.markdown(f"**Preview of {first_data_file.name} with selected columns:**")
                    preview_selected = preview_df[[rrn_column, amount_column]].head()
                    st.dataframe(preview_selected, width="expand")
                    
                except Exception as e:
                    st.warning(f"Could not preview {first_data_file.name}: {str(e)}")
                    
                    # Fallback manual input
                    col_a, col_b = st.columns(2)
                    with col_a:
                        rrn_column = st.text_input("RRN Column Name:", value="rrn")
                    with col_b:
                        amount_column = st.text_input("Amount Column Name:", value="amount")

    # Process button
    st.markdown("---")
    st.markdown("## 🔄 Start Verification Process")
    
    # Add reset button for verified database
    col_reset, col_start = st.columns([1, 3])
    
    with col_reset:
        if st.button("🗑️ Reset Database", help="Clear previously verified transaction IDs to avoid duplicates"):
            if os.path.exists("verified_ID.csv"):
                os.remove("verified_ID.csv")
                st.success("✅ Verified ID database cleared!")
            else:
                st.info("ℹ️ No database to clear")
    
    with col_start:
        start_verification = st.button("🚀 Start Verification", type="primary", width="expand")
        
    if start_verification:
        if st.session_state.processing:
            st.info("A verification run is already in progress. Please wait...")
            st.stop()
        st.session_state.processing = True
        if uploaded_csv is not None and uploaded_files:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Save files
                status_text.text("📁 Saving uploaded files...")
                progress_bar.progress(10)
                
                # Save registration file with appropriate extension
                if uploaded_csv.name.endswith('.xlsx'):
                    with open("input.xlsx", "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                else:
                    with open("input.csv", "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                
                # Save transaction report files
                if uploaded_files:
                    # Save all files with numbered suffixes if multiple
                    for i, file in enumerate(uploaded_files):
                        if file.name.endswith('.xlsx'):
                            filename = f"TransactionReport_{i}.xlsx" if len(uploaded_files) > 1 else "TransactionReport.xlsx"
                            with open(filename, "wb") as f:
                                f.write(file.getbuffer())
                        elif file.name.endswith('.csv'):
                            filename = f"TransactionReport_{i}.csv" if len(uploaded_files) > 1 else "TransactionReport.csv"
                            with open(filename, "wb") as f:
                                f.write(file.getbuffer())
                    
                    # Save column configuration for CSV/Excel files
                    if rrn_column and amount_column:
                        column_config = {
                            "rrn_column": rrn_column,
                            "amount_column": amount_column
                        }
                        import json
                        with open("column_config.json", "w") as f:
                            json.dump(column_config, f)
                
                progress_bar.progress(20)
                
                # Step 2: Extract transaction IDs
                status_text.text("🔍 Extracting transaction IDs from screenshots...")
                progress_bar.progress(40)
                
                extraction.main()
                
                # Step 3: Verify IDs
                status_text.text("✅ Verifying transaction IDs...")
                progress_bar.progress(70)
                
                ID_verify.main()
                
                # Step 4: Complete
                status_text.text("🎉 Verification process completed!")
                progress_bar.progress(100)
                
                st.success("✅ Verification process completed successfully!")
                
                # Display results
                if os.path.exists("verified_transactions.csv"):
                    st.markdown("## 📊 Verification Results")
                    
                    # Load and display results
                    df_results = pd.read_csv("verified_transactions.csv")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_records = len(df_results)
                        st.metric("Total Records", total_records)
                    
                    with col2:
                        verified_count = len(df_results[df_results['Verification'] == 'Verified'])
                        st.metric("Verified", verified_count)
                    
                    with col3:
                        not_verified_count = len(df_results[df_results['Verification'] == 'Not Verified'])
                        st.metric("Not Verified", not_verified_count)
                    
                    with col4:
                        no_id_count = len(df_results[df_results['Verification'] == 'No ID extracted'])
                        st.metric("No ID Extracted", no_id_count)
                    
                    # Results table
                    st.markdown("### Detailed Results")
                    st.dataframe(df_results, width="expand")
                    
                    # Download button
                    with open("verified_transactions.csv", "rb") as file:
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=file,
                            file_name=f"verified_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width="expand"
                        )
                else:
                    st.error("❌ The output file 'verified_transactions.csv' was not generated.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
            finally:
                st.session_state.processing = False
        else:
            st.warning("⚠️ Please upload both the registration file and at least one transaction report file to proceed.")
            st.session_state.processing = False

elif page == "📊 Results":
    st.markdown("## 📊 Previous Results")
    
    if os.path.exists("verified_transactions.csv"):
        df_results = pd.read_csv("verified_transactions.csv")
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            verified_pct = (len(df_results[df_results['Verification'] == 'Verified']) / len(df_results)) * 100
            st.metric("Verification Rate", f"{verified_pct:.1f}%")
        
        with col2:
            total_records = len(df_results)
            st.metric("Total Records", total_records)
        
        with col3:
            verified_count = len(df_results[df_results['Verification'] == 'Verified'])
            st.metric("Successfully Verified", verified_count)
        
        # Filter options
        st.markdown("### 🔍 Filter Results")
        filter_option = st.selectbox(
            "Filter by verification status:",
            ["All", "Verified", "Not Verified", "No ID extracted"]
        )
        
        if filter_option != "All":
            filtered_df = df_results[df_results['Verification'] == filter_option]
        else:
            filtered_df = df_results
        
        st.dataframe(filtered_df, width="expand")
        
        # Download filtered results
        if filter_option != "All":
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {filter_option} Results",
                data=csv_data,
                file_name=f"{filter_option.lower().replace(' ', '_')}_results.csv",
                mime="text/csv"
            )
    else:
        st.info("📝 No previous results found. Run the verification process first.")

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About Payment Verification OCR")
    
    st.markdown("""
    ### 🎯 Purpose
    This system automates the verification of payment transactions using advanced OCR (Optical Character Recognition) 
    and AI-powered object detection to extract and validate transaction IDs from payment screenshots.
    
    ### 🔧 How It Works
    1. **Upload Registration Data**: CSV or Excel file containing user information and screenshot URLs
    2. **Upload Transaction Reports**: CSV or Excel files with official transaction records
    3. **AI Processing**: 
       - Downloads screenshots from URLs
       - Uses YOLO model to detect transaction ID regions (if available)
       - Falls back to full-image OCR if no model is available
       - Extracts transaction IDs using OCR
    4. **Verification**: Matches extracted IDs with official transaction records
    5. **Results**: Generates comprehensive verification report
    
    ### 🛠️ Technology Stack
    - **Python**: Core programming language
    - **YOLOv12**: Object detection for cropping relevant regions
    - **pytesseract**: OCR for text extraction
    - **Streamlit**: Web interface
    - **Pandas**: Data processing
    - **OpenCV**: Image processing
    
    ### 📋 Supported Payment Platforms
    - **PhonePe**: UTR numbers (T + 21 digits)
    - **Google Pay**: Transaction IDs (e.g., AXIS1234567890)
    - **Paytm**: Reference numbers (12-15 digits)
    - **Amazon Pay**: Bank Reference IDs
    
    ### 📁 File Requirements
    - **Input CSV/Excel**: Must contain 'screenshots' column with image URLs (Max 200MB per file)
    - **Transaction Reports**: CSV or Excel files with official transaction reports for verification (Max 200MB per file)
    """)
    
    # System information
    st.markdown("### 💻 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required Files:**")
        model_exists = os.path.exists("model.pt")
        st.write(f"YOLO Model: {'✅ Present' if model_exists else '❌ Missing'}")
        
        if os.path.exists("input.csv") or os.path.exists("input.xlsx"):
            st.write("Input File: ✅ Present")
        else:
            st.write("Input File: ❌ Not uploaded")
            
        # Check for transaction report files (single or multiple)
        transaction_files = []
        for ext in ["csv", "xlsx"]:
            if os.path.exists(f"TransactionReport.{ext}"):
                transaction_files.append(f"TransactionReport.{ext}")
            # Check for numbered files
            numbered = glob.glob(f"TransactionReport_*.{ext}")
            transaction_files.extend(numbered)
        
        if transaction_files:
            st.write(f"Transaction Report: ✅ Present ({len(transaction_files)} file(s))")
        else:
            st.write("Transaction Report: ❌ Not uploaded")
    
    with col2:
        st.markdown("**Output Files:**")
        if os.path.exists("verified_transactions.csv"):
            st.write("Results CSV: ✅ Generated")
        else:
            st.write("Results CSV: ❌ Not generated")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Payment Verification OCR System | Built with Streamlit and AI"
    "</div>",
    unsafe_allow_html=True
)

