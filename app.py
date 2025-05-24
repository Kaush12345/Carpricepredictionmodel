import streamlit as st
import pandas as pd
import os
import sqlite3
from io import BytesIO
from dotenv import load_dotenv
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import smtplib
from email.message import EmailMessage
import traceback
from datetime import datetime
from user_db import create_user_table, register_user, authenticate_user, verify_user_email, send_email_verification, \
    request_password_reset, reset_password, create_model_table, save_model_metadata, get_user_models, get_model_path, \
    delete_model_metadata, check_admin_status
from streamlit import session_state
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import tempfile

# Load environment variables from .env
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS", "")

# --- Database Table Creation Functions (Moved to Top) ---
def create_prediction_log_table():
    conn = sqlite3.connect("prediction_log.db")
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS history
                   (
                       timestamp TEXT,
                       username TEXT,
                       data TEXT
                   )
                   ''')
    conn.commit()
    conn.close()

def create_preferences_table():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS user_preferences
                   (
                       user_id
                       INTEGER
                       PRIMARY
                       KEY,
                       email_notifications
                       BOOLEAN
                       DEFAULT
                       1,
                       model_updates
                       BOOLEAN
                       DEFAULT
                       1,
                       theme
                       TEXT
                       DEFAULT
                       'Dark',
                       currency
                       TEXT
                       DEFAULT
                       'INR',
                       FOREIGN
                       KEY
                   (                       user_id
                   ) REFERENCES users
                   (                       id
                   )
                       )                    ''')
    conn.commit()
    conn.close()

# --- Initial Database Setup --- 
# Create user and model tables if they don't exist (These are imported from user_db.py)
create_user_table()
create_model_table()

# Create other necessary tables if they don't exist (Definitions are now above)
create_prediction_log_table()
create_preferences_table()

st.set_page_config(page_title="Car Price Predictor", layout="centered")

# --- Custom CSS for modern dark theme and better UI ---
st.markdown('''
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
        color: #f5f6fa !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .stApp {
        padding: 0 !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
        background: rgba(30,32,34,0.95);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .stTitle, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f5f6fa !important;
        text-align: center;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        font-size: 1.1em;
        font-weight: 600;
        margin: 0.5em 0;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #dd2476 0%, #ff512f 100%) !important;
        color: #fff !important;
        transform: scale(1.04);
    }
    .stTextInput > div > input, .stTextArea > div > textarea, .stSelectbox > div {
        background: #232526 !important;
        color: #f5f6fa !important;
        border-radius: 8px;
        border: 1px solid #444;
        font-size: 1.1em;
        margin-bottom: 0.5em;
    }
    .stSidebar {
        background: #18191a !important;
        color: #f5f6fa !important;
        border-radius: 0 18px 18px 0;
        box-shadow: 2px 0 16px 0 rgba(31, 38, 135, 0.17);
    }
    .stSidebar .stMarkdown h2 {
        color: #ff512f !important;
        text-align: left;
        font-size: 1.3em;
        margin-bottom: 0.5em;
    }
    .stDataFrame, .stTable {
        background: #232526 !important;
        color: #f5f6fa !important;
        border-radius: 8px;
        margin-bottom: 1em;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #36d1c4 0%, #1fa2ff 100%) !important;
        color: #fff !important;
        border-radius: 8px;
        font-weight: 600;
        margin: 0.5em 0;
    }
    .stExpanderHeader {
        color: #ff512f !important;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 8px;
    }
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
            max-width: 100%;
        }
        .stButton > button {
            width: 100%;
        }
        .stTextInput > div > input, .stTextArea > div > textarea, .stSelectbox > div {
            width: 100%;
        }
    }
    </style>
''', unsafe_allow_html=True)

# --- Sidebar Branding ---
st.sidebar.markdown("""
<h2>🚗 Car Predictor</h2>
<p style='color:#aaa;font-size:0.95em;'>by Kaushik</p>
<hr style='border:1px solid #333;'>
""", unsafe_allow_html=True)

st.title("🚗 Car Price Predictor")


def hide_sidebar_if_needed():
    if (
            session_state.get('forgot_password', False)
            or session_state.get('verification_pending', False)
    ):
        st.markdown(
            """
            <style>
            [data-testid='stSidebar'] {display: none;}
            </style>
            """,
            unsafe_allow_html=True,
        )


hide_sidebar_if_needed()


# -------------------- Utility Functions --------------------
def send_model_training_notification(email, model_name, rmse):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return False
    msg = EmailMessage()
    msg['Subject'] = f'Model Training Complete: {model_name}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    msg.set_content(f'Your model "{model_name}" has been trained successfully. RMSE: ₹{rmse:,.2f}')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def preprocess_input(df, existing_label_encoders=None):
    """
    Preprocess input data for prediction, handling categorical and numerical features.
    Preserves feature names and handles unseen categories by mapping to a placeholder.
    """
    df = df.copy()

    # Define feature categories and their expected types
    categorical_feature_names = [
        'make', 'model', 'body_style', 'color', 'transmission_type',
        'fuel_type', 'drivetrain', 'condition_rating', 'service_records',
        'vehicle_history', 'location', 'season', 'listing_platform',
        'seller_type', 'safety_features', 'entertainment_system'
    ]

    # Clean string/object columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Identify columns to be label encoded
    cols_to_encode = [col for col in df.columns if df[col].dtype == object or col in categorical_feature_names]
    cols_to_encode = [col for col in cols_to_encode if col in df.columns]

    label_encoders = existing_label_encoders if existing_label_encoders else {}

    print("--- Preprocessing --- Identifying columns to encode:", cols_to_encode)

    # Process categorical columns
    for col in cols_to_encode:
        if col in df.columns:
            # Fill missing values before encoding with a string placeholder
            df[col] = df[col].fillna("Missing").astype(str)  # Fill with string, ensure string type

            if col in label_encoders:
                le = label_encoders[col]
                try:
                    # Attempt to transform using the existing encoder
                    # Unseen labels will cause a ValueError
                    encoded_data = le.transform(df[col])
                    df[col] = pd.Series(encoded_data, index=df.index, dtype=np.int64)
                    print(
                        f"Encoded column {col}. Example original: {df[col].iloc[0]}, Encoded: {df[col].iloc[0]}, Dtype: {df[col].dtype}")
                except ValueError as e:
                    print(f"ValueError transforming column {col}: {e}. Contains unseen labels. Setting to -1.")
                    # Handle unseen labels by mapping them to a placeholder value (e.g., -1)
                    # Create a mapping from known classes
                    mapping = {label: i for i, label in enumerate(le.classes_)}
                    # Map existing values, set unseen to -1
                    df[col] = df[col].map(mapping).fillna(-1).astype(np.int64)
                    st.warning(f"Column '{col}' contains unseen categories. Mapping to a placeholder.")
                except Exception as e:
                    print(f"Error transforming column {col}: {e}. Setting to -1.")
                    st.warning(f"Could not transform column '{col}': {e}. Setting to -1.")
                    df.loc[:, col] = -1  # Use a placeholder like -1 for transformation failures
            else:  # This block is primarily for training when no existing encoder is provided
                le = LabelEncoder()
                # Ensure there's data to fit on and it's string type
                if not df[col].empty and df[col].dtype == object:
                    encoded_data = le.fit_transform(df[col].astype(str))
                    df[col] = pd.Series(encoded_data, index=df.index, dtype=np.int64)
                    label_encoders[col] = le
                    print(
                        f"Fitted and encoded column {col}. Example original: {df[col].iloc[0]}, Encoded: {df[col].iloc[0]}, Dtype: {df[col].dtype}")
                else:
                    st.warning(f"Column '{col}' is empty or not object type, cannot fit LabelEncoder. Setting to -1.")
                    df.loc[:, col] = -1
        else:
            # Handle case where a column expected to be encoded is not in the input DataFrame
            if col in (existing_label_encoders if existing_label_encoders else {}):
                st.warning(f"Column '{col}' is missing from input data but was in the training model. Filling with -1.")
                df[col] = -1

    # Handle numerical columns
    numerical_columns = [
        'year', 'mileage', 'engine_size', 'horsepower', 'torque', 'fuel_efficiency',
        'cylinders', 'previous_owners', 'odometer', 'market_demand', 'economic_indicator',
        'fuel_prices', 'days_listed', 'views', 'inquiries'
    ]

    num_cols_present = [col for col in numerical_columns if col in df.columns]

    # Convert to numeric and handle errors
    for col in num_cols_present:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values in numerical columns
    num_cols_for_imputation = df[num_cols_present].select_dtypes(include=['int64', 'float64']).columns

    if len(num_cols_for_imputation) > 0:
        if df[num_cols_for_imputation].isnull().values.any():
            # Create a DataFrame with only the numerical columns to preserve feature names
            num_df = df[num_cols_for_imputation].copy()
            imputer = SimpleImputer(strategy="mean")
            # Fit and transform while preserving column names
            imputed_data = imputer.fit_transform(num_df)
            df[num_cols_for_imputation] = pd.DataFrame(
                imputed_data,
                columns=num_cols_for_imputation,
                index=df.index
            )
            print("Imputed missing values in numerical columns:", num_cols_for_imputation.tolist())
        else:
            print("No missing numerical values to impute.")

    # Handle boolean columns
    boolean_columns = [
        'accident_history', 'warranty_status', 'sunroof', 'navigation',
        'leather_seats', 'parking_sensors', 'backup_camera'
    ]

    for col in boolean_columns:
        if col in df.columns:
            # Convert to string first to handle mixed types
            df[col] = df[col].astype(str).str.title().map(
                {'Yes': 1, 'True': 1, '1': 1, 'No': 0, 'False': 0, '0': 0, np.nan: 0}
            ).fillna(0)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            print(f"Processed boolean column {col}. Example: {df[col].iloc[0]}, Dtype: {df[col].dtype}")

    # Ensure all columns used by the model are present and in the correct order
    if existing_label_encoders is not None and hasattr(existing_label_encoders, 'model_features'):
        expected_features = existing_label_encoders.model_features
        current_features = df.columns.tolist()

        # Add missing features with default values
        missing_features = [feat for feat in expected_features if feat not in current_features]
        if missing_features:
            print(f"Warning: Missing features after preprocessing: {missing_features}. Adding with default 0.")
            for feat in missing_features:
                # Use -1 for missing *encoded* categorical features, 0 for numerical
                if feat in (label_encoders if label_encoders else {}):
                    df[feat] = -1
                else:
                    df[feat] = 0

        # Ensure order matches the training data features
        df = df[expected_features].copy()

        # Final type check and conversion - ensure encoded columns are int64
        for feat in df.columns:
            if feat in (label_encoders if label_encoders else {}):  # Check if the feature was meant to be encoded
                if df[feat].dtype != np.int64:
                    print(f"Final check: Encoded column {feat} is not int64. Converting.")
                    # Coerce errors during conversion to handle unexpected values, fill with -1
                    df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(-1).astype(np.int64)

    print("--- Preprocessing Complete --- Final DataFrame dtypes:\n", df.dtypes)
    return df, label_encoders


def get_user_id(username):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def train_and_save_model(df, model_name="model"):
    df = df.copy()

    target_candidates = ['selling_price', 'price', 'target', 'car_price']
    target_column = next((col for col in df.columns if str(col).strip().lower() in target_candidates), None)
    if not target_column:
        raise ValueError(
            "No suitable target column found. Please include a column like 'Selling_Price', 'Price', or 'Target'.")

    # Convert target column to numeric, coercing errors to NaN
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Drop rows where the target column is NaN
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    remaining_rows = len(df)

    # --- Add a check here ---
    if remaining_rows == 0:
        raise ValueError(
            f"No valid data left after dropping rows with missing values in the target column ('{target_column}'). "
            f"Please ensure your dataset includes the '{target_column}' column with at least one non-missing value."
        )
    # --- End of check ---

    y = df[target_column]

    X = df.drop(target_column, axis=1, errors='ignore')

    X_processed, label_encoders = preprocess_input(X)

    # Ensure X_processed is not empty before splitting
    if len(X_processed) == 0:
        raise ValueError(
            f"No features left after preprocessing. "
            f"Please check your dataset and preprocessing logic."
        )

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Define model_features here before using it
    model_features = list(X_processed.columns)

    # Save model metadata
    user_id = get_user_id(st.session_state.get('username'))
    if user_id:
        # Make sure model_name is unique for the user
        if get_model_path(user_id, model_name):
            st.warning(f"Model name '{model_name}' already exists. Please use a different name to save.")
        else:
            model_file_path = f"models/{user_id}_{model_name}.pkl"
            # Ensure models directory exists
            os.makedirs('models', exist_ok=True)
            # Use the defined model_features variable in joblib.dump
            joblib.dump((model, label_encoders, target_column, model_features), model_file_path)
            save_model_metadata(user_id, model_name, model_file_path, rmse)
            st.success(f"Model trained and saved as '{model_name}'. RMSE: ₹{rmse:,.2f}")
            # Send email notification
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
            user_email = cursor.fetchone()[0]
            conn.close()
            if send_model_training_notification(user_email, model_name, rmse):
                st.info("A notification email has been sent to your registered email.")
            # Use the defined model_features variable here
            st.session_state['loaded_model'] = model
            st.session_state['loaded_label_encoders'] = label_encoders
            st.session_state['loaded_target_column'] = target_column
            st.session_state['loaded_model_features'] = model_features
            st.session_state['loaded_model_name'] = model_name
    else:
        st.warning("Model trained, but cannot be saved. Please log in to save models.")

    return rmse


def load_model(model_name="model", file_path=None):
    if file_path:
        model_path = file_path
    else:
        model_path = f"{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please upload a CSV to train the model.")
    model_data = joblib.load(model_path)
    if len(model_data) == 4:
        return model_data
    elif len(model_data) == 3:
        model, label_encoders, target_column = model_data
        return model, label_encoders, target_column, []
    else:
        # If model format is invalid, delete the file to prevent future errors
        os.remove(model_path)
        raise ValueError("Invalid model format. Please upload a valid CSV to retrain the model.")


def decode_labels(df, label_encoders):
    """
    Decode integer-encoded categorical labels back to their original string values.
    Handles potential errors and missing values gracefully.
    """
    df = df.copy()

    for col in df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Create a direct mapping dictionary from integer index to class name
            decoding_dict = {i: label for i, label in enumerate(le.classes_)}

            # Ensure the column is numeric (it should be after preprocessing)
            # Coerce errors to NaN, which will be handled below
            numeric_col = pd.to_numeric(df[col], errors='coerce')

            # --- Debug: Inside decode_labels ---
            print(f"--- Debug: Decoding column: {col} ---")
            print(f"Numeric column head:\n{numeric_col.head().to_string()}")
            print(f"Decoding dictionary for {col}:\n{decoding_dict}")
            print("---")
            # --- End Debug ---

            # Map the numeric values using the decoding dictionary
            # .map() will replace unknown integers or NaNs with NaN
            mapped_values = numeric_col.map(decoding_dict)

            # Fill any remaining NaNs (from coercion, missing original values, or unmapped integers) with 'Unknown/Unseen'
            df[col] = mapped_values.fillna("Unknown/Unseen").astype(object)

    return df


class EnhancedFPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Car Price Prediction Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def add_dataframe(self, df):
        # Basic table drawing for DataFrame
        if df.empty:
            self.set_font('Arial', '', 10)
            self.cell(0, 10, 'No data available.', 0, 1)
            self.ln(5)
            return
        self.set_font('Arial', 'B', 10)
        # Headers
        for col in df.columns:
            self.cell(40, 7, str(col), 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 10)
        # Data rows
        for index, row in df.iterrows():
            for col in df.columns:
                # Handle potential non-string data for PDF cell
                cell_value = str(row[col]) if pd.notnull(row[col]) else ""
                # Basic truncation if cell too wide
                if len(cell_value) > 30: cell_value = cell_value[:27] + "..."
                self.cell(40, 7, cell_value, 1, 0, 'L')
            self.ln()
        self.ln(5)

    def add_matplotlib_plot(self, fig):
        # Save the figure to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig.savefig(tmpfile, format='png')
            tmp_path = tmpfile.name

        # Add the image from the temporary file
        try:
            self.image(tmp_path, x=self.get_x(), y=self.get_y(), w=180)
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)

        self.ln(80)  # Adjust space after image


def generate_enhanced_pdf(dataframe, model, model_features, original_df=None, target_column=None):
    pdf = EnhancedFPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 10)

    # --- Prediction Results Table ---
    # Ensure we use the original data provided by the user for display in the results table,
    # combined with the predicted price.

    # Create a DataFrame for PDF display
    if original_df is not None:
        # Use original_df as the base for the display DataFrame
        pdf_display_df = original_df.copy()
    else:
        # If original_df is not available, create a DataFrame with the same index as the prediction results
        pdf_display_df = pd.DataFrame(index=dataframe.index)

    # Add the predicted price from the prediction results dataframe, ensuring index alignment
    if 'Predicted_Price' in dataframe.columns:
        pdf_display_df['Predicted_Price'] = dataframe['Predicted_Price'].reindex(pdf_display_df.index)

    # Determine which columns to display in the PDF table
    # Start with columns from the PDF display DataFrame
    display_cols = list(pdf_display_df.columns)

    # Exclude the original target column if it's present and different from Predicted_Price
    if target_column in display_cols and target_column != 'Predicted_Price':
        display_cols.remove(target_column)

    # Optionally, move Predicted_Price to the end of the list for better readability
    if 'Predicted_Price' in display_cols and display_cols[-1] != 'Predicted_Price':
        display_cols.remove('Predicted_Price')
        display_cols.append('Predicted_Price')

    # Limit columns for display in PDF table if there are too many features
    if len(display_cols) > 15:  # Arbitrary limit to keep table readable
        # Prioritize original-like columns and then Predicted_Price
        cols_to_keep = [col for col in display_cols if col != 'Predicted_Price'][:14]
        if 'Predicted_Price' in display_cols:
            cols_to_keep.append('Predicted_Price')
        display_cols = cols_to_keep

        st.warning(f"Too many features to display all in the PDF table. Showing {len(display_cols)} relevant columns.")

    # Format columns for display in PDF (handle potential NaN/None)
    # Apply formatting directly to pdf_display_df
    for col in display_cols:
        if col != 'Predicted_Price':
            # Format feature columns: convert to string, show '-' for missing
            if col in pdf_display_df.columns:
                pdf_display_df[col] = pdf_display_df[col].apply(lambda x: str(x) if pd.notna(x) else "-")
            else:
                # Column somehow missing in pdf_display_df, fill with '-' as it's missing
                pdf_display_df[col] = "-"
        elif col == 'Predicted_Price' and col in pdf_display_df.columns:
            # Format predicted price as currency, show '-' for missing prediction
            pdf_display_df[col] = pdf_display_df[col].apply(lambda x: f'INR {x:,.2f}' if pd.notna(x) else "-")
        else:
            # Predicted_Price column somehow missing, fill with '-'
            pdf_display_df[col] = "-"

    pdf.chapter_title('Prediction Results')
    # Pass the correctly formatted DataFrame to add_dataframe
    pdf.add_dataframe(pdf_display_df[display_cols])

    # --- Feature Importance Plot ---
    if hasattr(model, 'feature_importances_'):
        pdf.chapter_title('Feature Importances')
        fig, ax = plt.subplots(figsize=(7, 3))
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        top_n = min(10, len(model_features))
        ax.barh([model_features[i] for i in sorted_idx[:top_n]][::-1], importances[sorted_idx[:top_n]][::-1],
                color='#ff512f')
        ax.set_xlabel('Importance')
        ax.set_title('Top Feature Importances')
        plt.tight_layout()
        pdf.add_matplotlib_plot(fig)
        plt.close(fig)  # Close the figure to free memory

    # Why this price explanations and comparisons (simplified for PDF)
    if not dataframe.empty:
        pdf.chapter_title('Why This Price?')
        # Note: Generating detailed explanations and comparisons for every row in a large PDF can be slow
        # We'll include a simplified summary or focus on a few examples if needed
        # For now, we'll add the individual explanations and comparisons directly if only a few predictions
        if len(dataframe) <= 10:  # Limit detailed explanations for brevity in PDF
            for idx, row in dataframe.iterrows():
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 10, f"Car {idx + 1}:", 0, 1)
                pdf.set_font('Arial', '', 10)
                explanation = explain_prediction(row, dataframe.columns)
                pdf.multi_cell(0, 5, explanation)
                # Simplified comparison - just show the similar car if found
                if original_df is not None:
                    similar = find_most_similar_car(original_df, original_df.loc[idx], original_df.columns)
                    if not similar.equals(original_df.loc[idx]):
                        pdf.multi_cell(0, 5, "Most similar car:")
                        # Create comparison dataframe with relevant columns and predicted price
                        # Select relevant columns, ensuring they exist in both dataframes
                        compare_cols = [col for col in final_results_df.columns if col != 'Predicted_Price']
                        # Add 'Predicted_Price' if it exists in the final_results_df
                        if 'Predicted_Price' in final_results_df.columns:
                            compare_cols.append('Predicted_Price')

                        # Ensure selected columns are present in both row and similar dataframes
                        compare_cols = [col for col in compare_cols if col in row.index and col in similar.index]

                        # Create a dictionary for the comparison DataFrame, explicitly converting values to string
                        comp_data = {}
                        comp_data['Feature'] = compare_cols
                        comp_data['Your Car'] = [str(row[col]) if pd.notna(row[col]) else "-" for col in compare_cols]
                        comp_data['Similar Car'] = [str(similar[col]) if pd.notna(similar[col]) else "-" for col
                                                    in compare_cols]

                        comp = pd.DataFrame(comp_data)

                        # Reorder columns for better readability
                        comp = comp[['Feature', 'Your Car', 'Similar Car']]

                        # Format Predicted_Price row if it exists
                        if 'Predicted_Price' in compare_cols:
                            pred_price_row_index = comp[comp['Feature'] == 'Predicted_Price'].index
                            if not pred_price_row_index.empty:
                                idx = pred_price_row_index[0]
                                try:
                                    your_price = float(row['Predicted_Price']) if pd.notna(
                                        row['Predicted_Price']) else np.nan
                                    similar_price = float(similar['Predicted_Price']) if pd.notna(
                                        similar['Predicted_Price']) else np.nan
                                    comp.loc[idx, 'Your Car'] = f'INR {your_price:,.2f}' if pd.notna(
                                        your_price) else "-"
                                    comp.loc[idx, 'Similar Car'] = f'INR {similar_price:,.2f}' if pd.notna(
                                        similar_price) else "-"
                                except ValueError:
                                    # Handle cases where conversion to float fails unexpectedly
                                    comp.loc[idx, 'Your Car'] = str(
                                        row['Predicted_Price'])  # Keep as string if conversion fails
                                    comp.loc[idx, 'Similar Car'] = str(
                                        similar['Predicted_Price'])  # Keep as string if conversion fails

                        pdf.add_dataframe(comp)
                        price_diff = row['Predicted_Price'] - similar['Predicted_Price']
                        pdf.multi_cell(0, 5, f"Price difference (Your Car - Similar Car): INR {price_diff:,.2f}")

    # Add other visualizations if data available
    if original_df is not None and target_column is not None and 'Predicted_Price' in dataframe.columns:
        # Price Distribution Plot
        if 'Predicted_Price' in dataframe.columns:
            pdf.chapter_title('Predicted Price Distribution')
            fig, ax = plt.subplots()
            ax.hist(dataframe['Predicted_Price'], bins=30, color='#1fa2ff')
            ax.set_xlabel('Predicted Price (INR)')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            pdf.add_matplotlib_plot(fig)
            plt.close(fig)

        # Correlation Heatmap Plot
        numeric_df = original_df[
            [f for f in model_features if f in original_df.columns] + [target_column]].select_dtypes(include=np.number)
        if 'Predicted_Price' in dataframe.columns:  # Add predicted price to correlation check if available
            predicted_price_col = dataframe['Predicted_Price']
            # Align indices before joining
            predicted_price_col.index = numeric_df.index
            numeric_df['Predicted_Price'] = predicted_price_col

        if not numeric_df.empty and len(numeric_df.columns) > 1:
            pdf.chapter_title('Feature Correlation Heatmap')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            plt.tight_layout()
            pdf.add_matplotlib_plot(fig)
            plt.close(fig)

        # Predicted vs. Actual Plot
        if target_column in original_df.columns:
            pdf.chapter_title('Predicted vs. Actual Price')
            fig, ax = plt.subplots()
            # Ensure original_df and dataframe indices are aligned for plotting
            actual_prices = original_df[target_column]
            predicted_prices = dataframe['Predicted_Price']
            # Align indices
            actual_prices.index = predicted_prices.index

            ax.scatter(actual_prices, predicted_prices, alpha=0.7, color='#36d1c4')
            ax.set_xlabel('Actual Price (INR)')
            ax.set_ylabel('Predicted Price (INR)')
            # Add ideal line only if there's data to plot
            if not actual_prices.empty:
                ax.plot([actual_prices.min(), actual_prices.max()], [actual_prices.min(), actual_prices.max()],
                        'r--')  # Ideal line
            plt.tight_layout()
            pdf.add_matplotlib_plot(fig)
            plt.close(fig)

    return pdf.output(dest='S').encode('latin1')


def send_prediction_email(recipient, dataframe, pdf_binary=None):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise ValueError("Email credentials not found in .env file.")

    msg = EmailMessage()
    msg['Subject'] = 'Car Price Prediction Results'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient
    msg.set_content('Please find attached your car price prediction result.')

    csv_data = dataframe.to_csv(index=False)
    msg.add_attachment(csv_data.encode(), maintype='application', subtype='csv', filename='prediction.csv')

    if pdf_binary:
        msg.add_attachment(pdf_binary, maintype='application', subtype='pdf', filename='prediction.pdf')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)


def reset_prediction_history():
    conn = sqlite3.connect("prediction_log.db")
    cursor = conn.cursor()
    # Changed from DROP TABLE to DELETE FROM to avoid conflicts
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()


def log_prediction(df):
    conn = sqlite3.connect("prediction_log.db")
    # Convert DataFrame to JSON string to store all columns flexibly
    df['Timestamp'] = datetime.now().isoformat()
    df['Username'] = st.session_state.get('username', 'anonymous')
    
    # Convert DataFrame to JSON string
    json_data = df.to_json(orient='records')
    
    # Insert the JSON data
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history (timestamp, username, data) VALUES (?, ?, ?)",
                  (df['Timestamp'].iloc[0], df['Username'].iloc[0], json_data))
    conn.commit()
    conn.close()


def get_prediction_history():
    conn = sqlite3.connect("prediction_log.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert stored JSON data back to DataFrame
    all_data = []
    for row in rows:
        timestamp, username, json_data = row
        df = pd.read_json(json_data)
        all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def show_feature_importance(model, model_features):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(7, 3))
        sorted_idx = importances.argsort()[::-1]
        top_n = min(10, len(model_features))
        ax.barh([model_features[i] for i in sorted_idx[:top_n]][::-1], importances[sorted_idx[:top_n]][::-1],
                color='#ff512f')
        ax.set_xlabel('Importance')
        ax.set_title('Top Feature Importances')
        st.pyplot(fig)
    else:
        st.info('Feature importance not available for this model.')


def explain_prediction(row, feature_names):
    summary = []

    # Vehicle Characteristics
    if 'year' in row:
        summary.append(f"Year: {row['year']}")
    if 'mileage' in row:
        summary.append(f"Mileage: {row['mileage']:,} km")
    if 'body_style' in row:
        summary.append(f"Body Style: {row['body_style']}")

    # Performance & Specifications
    if 'engine_size' in row:
        summary.append(f"Engine: {row['engine_size']}L")
    if 'horsepower' in row:
        summary.append(f"Power: {row['horsepower']} HP")
    if 'fuel_type' in row:
        summary.append(f"Fuel: {row['fuel_type']}")

    # Condition & History
    if 'previous_owners' in row:
        summary.append(f"Owners: {row['previous_owners']}")
    if 'condition_rating' in row:
        summary.append(f"Condition: {row['condition_rating']}")
    if 'accident_history' in row:
        summary.append(f"Accident History: {'Yes' if row['accident_history'] else 'No'}")

    # Optional Features
    premium_features = []
    if 'leather_seats' in row and row['leather_seats']:
        premium_features.append("Leather Seats")
    if 'sunroof' in row and row['sunroof']:
        premium_features.append("Sunroof")
    if 'navigation' in row and row['navigation']:
        premium_features.append("Navigation")
    if premium_features:
        summary.append(f"Premium Features: {', '.join(premium_features)}")

    # Market Factors
    if 'location' in row:
        summary.append(f"Location: {row['location']}")

    return ", ".join(summary)


def find_most_similar_car(df, target_row, feature_names):
    # If DataFrame is empty or has only one row, return the target row itself
    if df.empty or len(df) == 1:
        return target_row

    # Use only relevant features for similarity
    features = [f for f in feature_names if f != 'Predicted_Price' and f in df.columns]
    if not features:  # If no features are available, return target row
        return target_row

    df_comp = df[features].copy()
    target = target_row[features].copy()

    # Handle categorical features
    for col in df_comp.select_dtypes(include='object').columns:
        try:
            combined_series = pd.concat([df_comp[col].astype(str), pd.Series([str(target[col])])], axis=0).unique()
            le = LabelEncoder()
            le.fit(combined_series)
            df_comp[col] = le.transform(df_comp[col].astype(str))
            target[col] = le.transform([str(target[col])])[0]
        except Exception as e:
            print(f"Error encoding column {col}: {e}")
            # If encoding fails, drop the column
            df_comp = df_comp.drop(columns=[col])
            target = target.drop(labels=[col])

    # Ensure numerical types and handle missing values
    for col in df_comp.columns:
        try:
            df_comp[col] = pd.to_numeric(df_comp[col], errors='coerce')
            target[col] = pd.to_numeric(target[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col} to numeric: {e}")
            # If conversion fails, drop the column
            df_comp = df_comp.drop(columns=[col])
            target = target.drop(labels=[col])

    # If no columns left after processing, return target row
    if df_comp.empty:
        return target_row

    # Impute missing values
    try:
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_comp)
        df_comp_imputed = imputer.transform(df_comp)
        target_imputed = imputer.transform(target.values.reshape(1, -1))
    except Exception as e:
        print(f"Error during imputation: {e}")
        return target_row

    # Compute weighted Euclidean distance
    # Give more weight to important features
    weights = {
        'year': 2.0,
        'mileage': 2.0,
        'condition_rating': 1.5,
        'accident_history': 1.5,
        'engine_size': 1.2,
        'horsepower': 1.2,
        'price': 2.0
    }

    feature_weights = np.ones(len(features))
    for i, feat in enumerate(features):
        if feat in weights:
            feature_weights[i] = weights[feat]

    try:
        # Calculate weighted distances
        dists = np.sqrt(np.sum(((df_comp_imputed - target_imputed) * feature_weights) ** 2, axis=1))

        # Find the index of the target row to exclude it
        target_idx = -1
        for idx, row in df.iterrows():
            match = True
            for feat in df.columns:
                if pd.isna(row[feat]) and pd.isna(target_row[feat]):
                    continue
                if row[feat] != target_row[feat]:
                    match = False
                    break
            if match:
                target_idx = idx
                break

        if target_idx != -1:
            dists[target_idx] = np.inf

        idx = np.argmin(dists)
        return df.iloc[idx]
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return target_row


def plot_feature_vs_price(df, feature):
    if feature in df.columns:
        st.markdown(f"#### {feature} vs. Price")
        fig, ax = plt.subplots()
        ax.scatter(df[feature], df['Predicted_Price'], alpha=0.7, color='#36d1c4')
        ax.set_xlabel(feature)
        ax.set_ylabel('Predicted Price')
        st.pyplot(fig)


def plot_price_distribution(df):
    if 'Predicted_Price' in df.columns:
        st.markdown("#### Predicted Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Predicted_Price'], bins=30, color='#1fa2ff')
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)


def plot_correlation_heatmap(df, model_features):
    numeric_df = df[[f for f in model_features if f in df.columns] + ['Predicted_Price']].select_dtypes(
        include=np.number)
    if not numeric_df.empty:
        st.markdown("#### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)


def plot_predicted_vs_actual(df, target_column):
    if target_column in df.columns and 'Predicted_Price' in df.columns:
        st.markdown("#### Predicted vs. Actual Price")
        fig, ax = plt.subplots()
        ax.scatter(df[target_column], df['Predicted_Price'], alpha=0.7, color='#36d1c4')
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.plot([df[target_column].min(), df[target_column].max()], [df[target_column].min(), df[target_column].max()],
                'r--')  # Ideal line
        st.pyplot(fig)


# -------------------- User Authentication --------------------
# Add a state variable for forgot password
if 'forgot_password' not in st.session_state:
    st.session_state['forgot_password'] = False
if 'reset_email_sent' not in st.session_state:
    st.session_state['reset_email_sent'] = False
if 'reset_complete' not in st.session_state:
    st.session_state['reset_complete'] = False


def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    col1, col2 = st.columns([1, 2])
    login_clicked = col1.button("Login")
    forgot_clicked = col2.button("Forgot Password?")
    if login_clicked:
        success, msg = authenticate_user(username, password)
        if success:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['verification_pending'] = False
            st.success("Logged in successfully!")
        else:
            if msg == "Email not verified":
                st.session_state['verification_pending'] = True
                st.session_state['username'] = username
                st.warning("Email not verified. Please verify your email.")
            else:
                st.error(msg)
    if forgot_clicked:
        st.session_state['forgot_password'] = True
        st.session_state['reset_email_sent'] = False


def forgot_password_ui():
    st.subheader("Forgot Password")
    if st.session_state['reset_complete']:
        st.success("Your password has been reset. You can now log in with your new password.")
        if st.button("Back to Login", key="back_to_login_btn"):
            st.session_state['forgot_password'] = False
            st.session_state['reset_email_sent'] = False
            st.session_state['reset_complete'] = False
            st.session_state['reset_email'] = ""
        return
    if not st.session_state['reset_email_sent']:
        email = st.text_input("Enter your registered email", key="reset_email_input")
        if st.button("Send Reset Code", key="send_reset_code_btn"):
            sent, msg = request_password_reset(email, EMAIL_ADDRESS, EMAIL_PASSWORD)
            if sent:
                st.session_state['reset_email_sent'] = True
                st.session_state['reset_email'] = email
                st.success(msg)
            else:
                st.error(msg)
    else:
        code = st.text_input("Enter the reset code sent to your email", key="reset_code_input")
        new_password = st.text_input("Enter new password", type="password", key="reset_new_password_input")
        if st.button("Reset Password", key="reset_password_btn"):
            email = st.session_state.get('reset_email', None)
            if not email:
                st.error("Session error. Please try again.")
                return
            ok, msg = reset_password(email, code, new_password)
            if ok:
                st.session_state['reset_complete'] = True
                st.success(msg)
            else:
                st.error(msg)


def register():
    st.subheader("Register")
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    if st.button("Register"):
        success, response = register_user(username, password, email)
        if success:
            verification_code = response
            sent = send_email_verification(email, username, verification_code, EMAIL_ADDRESS, EMAIL_PASSWORD)
            if sent:
                st.session_state['verification_pending'] = True
                st.session_state['username'] = username
                st.success("User registered successfully! A verification code has been sent to your email.")
            else:
                st.error("User registered but failed to send verification email. Please contact support.")
        else:
            st.error(f"Registration failed: {response}")


def verify_email():
    st.subheader("Verify Email")
    username = st.session_state.get('username', None)
    if not username:
        st.error("No username found for verification. Please register or login first.")
        return
    verification_code = st.text_input("Enter verification code", key="verify_code")
    if st.button("Verify"):
        if verify_user_email(username, verification_code):
            st.success("Email verified successfully! You can now login.")
            st.session_state['verification_pending'] = False
        else:
            st.error("Invalid verification code. Please try again.")


# Move profile_page function before the main app section
def profile_page():
    st.header("👤 User Profile")

    # Create tabs for different profile sections
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Statistics", "Activity History", "Preferences"])

    username = st.session_state.get('username', '')

    # Fetch user info from DB
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT email, created_at FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    email = result[0] if result else ''
    created_at = result[1] if result else None  # Revert: Fetch created_at again

    # Fetch user statistics
    cursor.execute("SELECT COUNT(*) FROM models WHERE user_id = (SELECT id FROM users WHERE username = ?)", (username,))
    models_count = cursor.fetchone()[0]

    # Ensure the history table exists before querying it in the profile page
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            timestamp TEXT,
            username TEXT
            -- Add other columns here if you want a fixed schema
        )
    ''')
    conn.commit()

    # Get prediction count
    cursor.execute("SELECT COUNT(*) FROM history WHERE username = ?", (username,))
    prediction_count = cursor.fetchone()[0]

    # Get prediction history
    cursor.execute("SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC", (username,))

    # Get user preferences
    preferences = get_user_preferences(username)

    with tab1:
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            new_email = st.text_input("Email", value=email, key="profile_email")
            new_username = st.text_input("Username", value=username, key="profile_username")

            if st.button("Update Profile", key="update_profile_btn"):
                if not new_email or not new_username:
                    st.error("Email and username cannot be empty.")
                else:
                    try:
                        cursor.execute("UPDATE users SET email = ? WHERE username = ?", (new_email, username))
                        cursor.execute("UPDATE users SET username = ? WHERE username = ?", (new_username, username))
                        conn.commit()
                        st.session_state['username'] = new_username
                        st.success("Profile updated successfully!")
                    except Exception as e:
                        st.error(f"Update failed: {e}")

        with col2:
            st.info(f"Account created: {created_at}")
            st.info(f"Member since: {created_at.split()[0] if created_at else 'N/A'}")

        st.markdown("---")
        st.subheader("Change Password")
        col1, col2 = st.columns(2)
        with col1:
            old_pw = st.text_input("Current Password", type="password", key="old_pw")
            new_pw = st.text_input("New Password", type="password", key="new_pw")
            confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")

            if st.button("Change Password", key="change_pw_btn"):
                if not old_pw or not new_pw or not confirm_pw:
                    st.error("All password fields are required.")
                elif new_pw != confirm_pw:
                    st.error("New passwords do not match.")
                else:
                    from user_db import authenticate_user, hash_password
                    ok, _ = authenticate_user(username, old_pw)
                    if not ok:
                        st.error("Current password is incorrect.")
                    else:
                        pw_hash, salt = hash_password(new_pw)
                        cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                                       (pw_hash, salt, username))
                        conn.commit()
                        st.success("Password changed successfully!")

    with tab2:
        st.subheader("Your Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Trained", models_count)
        with col2:
            st.metric("Predictions Made", prediction_count)
        with col3:
            st.metric("Account Age",
                      f"{((datetime.now() - datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')).days if created_at else 0)} days")

        # Show model performance metrics
        if models_count > 0:
            st.subheader("Model Performance")
            cursor.execute("""
                           SELECT model_name, rmse, train_date
                           FROM models
                           WHERE user_id = (SELECT id FROM users WHERE username = ?)
                           ORDER BY train_date DESC
                           """, (username,))
            models = cursor.fetchall()

            if models:
                model_df = pd.DataFrame(models, columns=['Model Name', 'RMSE', 'Train Date'])
                st.dataframe(model_df)

                # Plot RMSE trend
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(model_df['Train Date'], model_df['RMSE'], marker='o')
                ax.set_xlabel('Training Date')
                ax.set_ylabel('RMSE')
                ax.set_title('Model Performance Trend')
                plt.xticks(rotation=45)
                st.pyplot(fig)

    with tab3:
        st.subheader("Recent Activity")

        # Show recent predictions
        st.write("Recent Predictions")
        cursor.execute("""
                       SELECT timestamp, COUNT (*) as prediction_count
                       FROM history
                       WHERE username = ?
                       GROUP BY DATE (timestamp)
                       ORDER BY timestamp DESC
                           LIMIT 10
                       """, (username,))
        recent_predictions = cursor.fetchall()

        if recent_predictions:
            pred_df = pd.DataFrame(recent_predictions, columns=['Date', 'Predictions'])
            st.dataframe(pred_df)

            # Plot prediction activity
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(pred_df['Date'], pred_df['Predictions'])
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Predictions')
            ax.set_title('Prediction Activity')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No prediction history available.")

    with tab4:
        st.subheader("Account Preferences")

        # Notification preferences
        st.write("Notification Settings")
        email_notifications = st.checkbox("Receive email notifications",
                                          value=preferences['email_notifications'])
        model_updates = st.checkbox("Receive model training updates",
                                    value=preferences['model_updates'])

        # Display preferences
        st.write("Display Settings")
        theme = st.selectbox("Theme",
                             ["Dark", "Light", "System Default"],
                             index=["Dark", "Light", "System Default"].index(preferences['theme']))
        currency = st.selectbox("Currency Display",
                                ["INR", "USD", "EUR"],
                                index=["INR", "USD", "EUR"].index(preferences['currency']))

        if st.button("Save Preferences"):
            new_preferences = {
                'email_notifications': email_notifications,
                'model_updates': model_updates,
                'theme': theme,
                'currency': currency
            }

            if update_user_preferences(username, new_preferences):
                st.success("Preferences saved successfully!")
                # Apply theme immediately
                if theme != "System Default":
                    apply_theme(theme)
            else:
                st.error("Failed to save preferences.")

    conn.close()


def get_user_preferences(username):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()

    # Get user_id
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return None

    user_id = result[0]

    # Get preferences
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    prefs = cursor.fetchone()

    if not prefs:
        # Create default preferences if none exist
        cursor.execute('''
                       INSERT INTO user_preferences (user_id, email_notifications, model_updates, theme, currency)
                       VALUES (?, 1, 1, 'Dark', 'INR')
                       ''', (user_id,))
        conn.commit()
        prefs = (user_id, 1, 1, 'Dark', 'INR')

    conn.close()
    return {
        'user_id': prefs[0],
        'email_notifications': bool(prefs[1]),
        'model_updates': bool(prefs[2]),
        'theme': prefs[3],
        'currency': prefs[4]
    }


def update_user_preferences(username, preferences):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()

    # Get user_id
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return False

    user_id = result[0]

    # Update preferences
    cursor.execute('''
                   UPDATE user_preferences
                   SET email_notifications = ?,
                       model_updates       = ?,
                       theme               = ?,
                       currency            = ?
                   WHERE user_id = ?
                   ''', (
                       preferences['email_notifications'],
                       preferences['model_updates'],
                       preferences['theme'],
                       preferences['currency'],
                       user_id
                   ))

    conn.commit()
    conn.close()
    return True


def apply_theme(theme):
    if theme == 'Dark':
        st.markdown('''
            <style>
            body, .stApp {
                background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
                color: #f5f6fa !important;
            }
            </style>
        ''', unsafe_allow_html=True)
    elif theme == 'Light':
        st.markdown('''
            <style>
            body, .stApp {
                background: #ffffff !important;
                color: #000000 !important;
            }
            </style>
        ''', unsafe_allow_html=True)


# Move model_management_page function before the main app section
def model_management_page():
    st.header("📊 Model Management")
    user_id = get_user_id(st.session_state.get('username'))  # Assuming you have a get_user_id function
    if not user_id:
        st.warning("Please log in to manage models.")
        return
    st.subheader("My Trained Models")
    models = get_user_models(user_id)
    if models:
        model_df = pd.DataFrame(models, columns=['Model Name', 'RMSE', 'Train Date'])
        st.dataframe(model_df)

        # Load Model
        st.subheader("Load a Model")
        model_names = [m[0] for m in models]
        selected_model_name = st.selectbox("Select a model to load:", ['--Select--'] + model_names,
                                           key="load_model_selectbox")
        if selected_model_name != '--Select--':
            if st.button("Load Model", key="load_model_button"):
                model_path = get_model_path(user_id, selected_model_name)
                if model_path and os.path.exists(model_path):
                    try:
                        model, label_encoders, target_column, model_features = joblib.load(model_path)
                        st.session_state['loaded_model'] = model
                        st.session_state['loaded_label_encoders'] = label_encoders
                        st.session_state['loaded_target_column'] = target_column
                        st.session_state['loaded_model_features'] = model_features
                        st.session_state['loaded_model_name'] = selected_model_name
                        st.success(f"Model '{selected_model_name}' loaded successfully.")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                else:
                    st.error(f"Model file not found for '{selected_model_name}'.")

        # Delete Model
        st.subheader("Delete a Model")
        selected_model_name_delete = st.selectbox("Select a model to delete:", ['--Select--'] + model_names,
                                                  key="delete_model_selectbox")
        if selected_model_name_delete != '--Select--':
            if st.button("Delete Model", key="delete_model_button"):
                model_path = get_model_path(user_id, selected_model_name_delete)
                if delete_model_metadata(user_id, selected_model_name_delete):
                    if model_path and os.path.exists(model_path):
                        os.remove(model_path)
                    st.success(f"Model '{selected_model_name_delete}' deleted.")
                    st.rerun()  # Rerun to update the list
                else:
                    st.error(f"Failed to delete model '{selected_model_name_delete}'.")

    else:
        st.info("You have not trained any models yet.")


# Move feedback_page function before the main app section
def feedback_page():
    st.header("📬 Feedback & Support")
    st.write("Have feedback or need support? Fill out the form below.")

    if not st.session_state.get('logged_in'):
        st.info("Please log in to submit feedback.")
        return

    username = st.session_state.get('username', 'Anonymous')

    with st.form(key='feedback_form'):
        subject = st.text_input("Subject")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button(label='Submit Feedback')

    if submit_button:
        if not subject or not message:
            st.warning("Please fill in both the subject and message.")
        else:
            # Here you would typically save the feedback to a database or send an email.
            # For now, we'll just show a success message.
            st.success("Thank you for your feedback!")
            # You can access the feedback here: subject, message, and username
            # print(f"Feedback from {username} (Subject: {subject}): {message}") # For debugging


# -------------------- Main App --------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'verification_pending' not in st.session_state:
    st.session_state['verification_pending'] = False

# Remove or comment out the debug print for session state
# st.write("DEBUG SESSION STATE:", dict(st.session_state))

if not st.session_state['logged_in']:
    if st.session_state['verification_pending']:
        verify_email()
        if st.button("Back to Login"):
            st.session_state['verification_pending'] = False
    elif st.session_state['forgot_password']:
        forgot_password_ui()
    else:
        choice = st.sidebar.selectbox("Choose Action", ["Login", "Register"])
        if choice == "Login":
            login()
        elif choice == "Register":
            register()
else:
    # Logged in user view
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    menu = ["Upload CSV", "Manual Input", "Prediction History", "Profile", "Model Management", "Logout"]
    username = st.session_state.get('username')
    if username and check_admin_status(username):
        menu.insert(5, "Admin Dashboard")  # Insert before Logout
    menu.insert(len(menu) - 1, "Feedback & Support")
    choice = st.sidebar.selectbox("Choose Action", menu)

    if choice == "Logout":
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.success("Logged out successfully!")

    elif choice == "Upload CSV":
        st.subheader("⬆️ Upload Car Data (for Training and Batch Prediction)")
        model_name = st.text_input("Model Name (for saving trained model)", value="model")
        uploaded_file = st.file_uploader("Upload your car dataset (.csv or .pdf)", type=["csv", "pdf"])
        retrain = st.checkbox("Retrain model with uploaded data", value=True)

        df = None
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
            elif uploaded_file.name.lower().endswith(".pdf"):
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        all_tables = []
                        for page in pdf.pages:
                            tables = page.extract_tables()
                            for table in tables:
                                # Convert table to DataFrame and append
                                df_table = pd.DataFrame(table[1:], columns=table[0])
                                all_tables.append(df_table)
                        if all_tables:
                            df = pd.concat(all_tables, ignore_index=True)
                        else:
                            st.error("No tables found in PDF.")
                except Exception as e:
                    st.error(f"Error reading PDF file: {e}")

            if df is not None and not df.empty:
                st.write("### Preview of Uploaded Data:")
                st.dataframe(df.head())

                # Store the original uploaded dataframe before any further processing for PDF report
                original_uploaded_df = df.copy()

                # Check if a model is loaded, if not, train or load default
                model = st.session_state.get('loaded_model', None)
                label_encoders = st.session_state.get('loaded_label_encoders', None)
                target_column = st.session_state.get('loaded_target_column', None)
                model_features = st.session_state.get('loaded_model_features', [])
                loaded_model_name = st.session_state.get('loaded_model_name', 'None')

                if retrain or not model:
                    if retrain:
                        st.info("Retraining model with uploaded data...")
                    else:
                        st.info("No model loaded, training a new one...")
                    try:
                        # train_and_save_model will also save metadata and load into session state if logged in
                        train_and_save_model(df.copy(), model_name)
                        # After training, model details are in session state if user is logged in
                        model = st.session_state.get('loaded_model', None)
                        label_encoders = st.session_state.get('loaded_label_encoders', None)
                        target_column = st.session_state.get('loaded_target_column', None)
                        model_features = st.session_state.get('loaded_model_features', [])
                        loaded_model_name = st.session_state.get('loaded_model_name', 'Newly Trained')
                    except Exception as e:
                        st.error(f"Error during model training: {e}")
                        st.text(traceback.format_exc())
                        model = None  # Ensure model is None if training fails

                if model:
                    st.subheader(f"🤖 Predicting Prices (using model: '{loaded_model_name}')")
                    try:
                        # Process the uploaded data (X) for prediction
                        if target_column in df.columns:
                            X = df.drop(target_column, axis=1, errors='ignore')
                        else:
                            X = df.copy()

                        # Ensure features match the model's expected features
                        for col in model_features:
                            if col not in X.columns:
                                X[col] = 0  # Add missing features with a default value
                        X = X[model_features]  # Select and reorder columns to match model features

                        processed_df, _ = preprocess_input(X, existing_label_encoders=label_encoders)

                        # Ensure features match the model's expected features
                        processed_df = processed_df[model_features].copy()  # Ensure order and only model features

                        predictions = model.predict(processed_df)

                        # --- Corrected Logic for Prediction Results and Decoding ---
                        # Add predictions to the processed_df (which has encoded categories)
                        processed_df['Predicted_Price'] = predictions

                        # Decode the labels in the processed_df
                        decoded_processed_df = decode_labels(processed_df.copy(), label_encoders)

                        # Merge decoded categorical columns back into the original dataframe (df)
                        # Identify categorical columns that were encoded
                        categorical_cols = [col for col in decoded_processed_df.columns if col in label_encoders]

                        # Create the final results dataframe by combining original columns with decoded categorical and prediction
                        # Start with original numerical and non-categorical columns
                        final_results_df = df.copy()

                        # Replace original categorical columns with decoded ones
                        for col in categorical_cols:
                            if col in final_results_df.columns and col in decoded_processed_df.columns:
                                # Ensure indices align when copying decoded column back
                                final_results_df[col] = decoded_processed_df[col].reindex(final_results_df.index)
                            elif col in decoded_processed_df.columns:
                                # Add decoded column if it wasn't originally present but appeared after preprocessing/encoding
                                final_results_df[col] = decoded_processed_df[col]

                        # Add the predicted price to the final results dataframe
                        if 'Predicted_Price' in decoded_processed_df.columns:
                            # Ensure indices align when copying predicted price back
                            final_results_df['Predicted_Price'] = decoded_processed_df['Predicted_Price'].reindex(
                                final_results_df.index)

                        # Use the final_results_df for logging, displaying, and downloading
                        log_prediction(final_results_df)  # Log the predictions

                        st.subheader("📊 Prediction Results:")
                        st.dataframe(final_results_df)
                        show_feature_importance(model, model_features)

                        # Explanation for each car (show for first few or sample if many)
                        st.markdown("---")
                        st.subheader("🔍 Why these prices?")

                        if len(final_results_df) > 50:  # Limit detailed explanations for large datasets
                            st.info(
                                f"Showing detailed explanation for the first 50 cars out of {len(final_results_df)}.")
                            display_df = final_results_df.head(50)
                        else:
                            display_df = final_results_df

                        for idx, row in display_df.iterrows():
                            st.markdown(f"**Car {row.name + 1}:** {explain_prediction(row, final_results_df.columns)}")
                            # Find and show most similar car (use original_df for similarity search)
                            # Ensure we use the original index to find the row in the original_df
                            original_row = df.loc[row.name]
                            similar = find_most_similar_car(final_results_df, row,
                                                            final_results_df.columns)  # Use final_results_df for similarity
                            if not similar.equals(row):
                                st.markdown("*Most similar car in your data:*")

                                # Create comparison dataframe with relevant columns and predicted price
                                # Select relevant columns, ensuring they exist in both dataframes
                                compare_cols = [col for col in final_results_df.columns if col != 'Predicted_Price']
                                # Add 'Predicted_Price' if it exists in the final_results_df
                                if 'Predicted_Price' in final_results_df.columns:
                                    compare_cols.append('Predicted_Price')

                                # Ensure selected columns are present in both row and similar dataframes
                                compare_cols = [col for col in compare_cols if
                                                col in row.index and col in similar.index]

                                # Create a dictionary for the comparison DataFrame, explicitly converting values to string
                                comp_data = {}
                                comp_data['Feature'] = compare_cols
                                comp_data['Your Car'] = [str(row[col]) if pd.notna(row[col]) else "-" for col in
                                                         compare_cols]
                                comp_data['Similar Car'] = [str(similar[col]) if pd.notna(similar[col]) else "-" for col
                                                            in compare_cols]

                                comp = pd.DataFrame(comp_data)

                                # Reorder columns for better readability
                                comp = comp[['Feature', 'Your Car', 'Similar Car']]

                                # Format Predicted_Price row if it exists
                                if 'Predicted_Price' in compare_cols:
                                    pred_price_row_index = comp[comp['Feature'] == 'Predicted_Price'].index
                                    if not pred_price_row_index.empty:
                                        idx = pred_price_row_index[0]
                                        try:
                                            your_price = float(row['Predicted_Price']) if pd.notna(
                                                row['Predicted_Price']) else np.nan
                                            similar_price = float(similar['Predicted_Price']) if pd.notna(
                                                similar['Predicted_Price']) else np.nan
                                            comp.loc[idx, 'Your Car'] = f'INR {your_price:,.2f}' if pd.notna(
                                                your_price) else "-"
                                            comp.loc[idx, 'Similar Car'] = f'INR {similar_price:,.2f}' if pd.notna(
                                                similar_price) else "-"
                                        except ValueError:
                                            # Handle cases where conversion to float fails unexpectedly
                                            comp.loc[idx, 'Your Car'] = str(
                                                row['Predicted_Price'])  # Keep as string if conversion fails
                                            comp.loc[idx, 'Similar Car'] = str(
                                                similar['Predicted_Price'])  # Keep as string if conversion fails

                                st.dataframe(comp)
                                price_diff = row['Predicted_Price'] - similar['Predicted_Price']
                                st.info(f"Price difference (Your Car - Similar Car): INR {price_diff:,.2f}")

                        # Visualizations for key features (use the processed data for plotting if needed)
                        st.markdown("---")
                        st.subheader("📈 Data Visualizations")
                        plot_price_distribution(final_results_df)
                        # Use final_results_df for correlation heatmap since it contains Predicted_Price
                        plot_correlation_heatmap(final_results_df, model_features)

                        # Plot predicted vs. actual only if target column is present in original df
                        if target_column in df.columns:
                            plot_predicted_vs_actual(df, target_column)

                        st.download_button("📥 Download CSV Results", final_results_df.to_csv(index=False),
                                           "prediction_results.csv", "text/csv")

                        # Generate and download Enhanced PDF Report
                        # Pass necessary data for PDF generation, using the original_uploaded_df
                        pdf_data = generate_enhanced_pdf(final_results_df, model, model_features, original_uploaded_df,
                                                         target_column)
                        st.download_button("📄 Download PDF Report", pdf_data, "prediction_report.pdf",
                                           "application/pdf")

                        with st.expander("📧 Email Results"):
                            email = st.text_input("Recipient Email")
                            if st.button("Send Email", key="send_email_upload"):
                                try:
                                    # send_prediction_email function already handles attaching PDF
                                    send_prediction_email(email, final_results_df, pdf_binary=pdf_data)
                                    st.success("Email sent successfully!")
                                except Exception as e:
                                    st.error(f"Email error: {e}")
                                    st.text(traceback.format_exc())

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.text(traceback.format_exc())

                else:
                    st.warning("Please train or load a model first.")

            else:
                st.info("Uploaded data is empty.")

    elif choice == "Manual Input":
        st.subheader("📌 Manual Input")
        model = st.session_state.get('loaded_model', None)
        label_encoders = st.session_state.get('loaded_label_encoders', None)
        target_column = st.session_state.get('loaded_target_column', None)
        model_features = st.session_state.get('loaded_model_features', [])
        loaded_model_name = st.session_state.get('loaded_model_name', 'None')

        if model:
            st.info(f"Using loaded model: '{loaded_model_name}'")

            # Create tabs for different feature categories
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Vehicle Characteristics", "Performance & Specs",
                "Condition & History", "Market Factors",
                "Optional Features", "Listing Info"
            ])

            input_data = {}

            with tab1:
                st.subheader("Vehicle Characteristics")
                col1, col2 = st.columns(2)
                with col1:
                    if 'make' in model_features:
                        input_data['make'] = st.text_input("Make", key="input_make")
                    if 'model' in model_features:
                        input_data['model'] = st.text_input("Model", key="input_model")
                    if 'year' in model_features:
                        input_data['year'] = st.number_input("Year", min_value=1900, max_value=2024, value=2020,
                                                             key="input_year")
                with col2:
                    if 'mileage' in model_features:
                        input_data['mileage'] = st.number_input("Mileage (km)", min_value=0, value=50000,
                                                                key="input_mileage")
                    if 'body_style' in model_features:
                        input_data['body_style'] = st.selectbox("Body Style",
                                                                ["Sedan", "SUV", "Hatchback", "Coupe", "Wagon", "Van",
                                                                 "Pickup"], key="input_body_style")
                    if 'color' in model_features:
                        input_data['color'] = st.text_input("Color", key="input_color")
                    if 'transmission_type' in model_features:
                        input_data['transmission_type'] = st.selectbox("Transmission",
                                                                       ["Automatic", "Manual", "CVT"],
                                                                       key="input_transmission")

            with tab2:
                st.subheader("Performance & Specifications")
                col1, col2 = st.columns(2)
                with col1:
                    if 'engine_size' in model_features:
                        input_data['engine_size'] = st.number_input("Engine Size (L)", min_value=0.0, value=2.0,
                                                                    key="input_engine_size")
                    if 'fuel_type' in model_features:
                        input_data['fuel_type'] = st.selectbox("Fuel Type",
                                                               ["Petrol", "Diesel", "Electric", "Hybrid"],
                                                               key="input_fuel_type")
                    if 'horsepower' in model_features:
                        input_data['horsepower'] = st.number_input("Horsepower", min_value=0, value=150,
                                                                   key="input_horsepower")
                with col2:
                    if 'torque' in model_features:
                        input_data['torque'] = st.number_input("Torque (Nm)", min_value=0, value=200,
                                                               key="input_torque")
                    if 'drivetrain' in model_features:
                        input_data['drivetrain'] = st.selectbox("Drivetrain",
                                                                ["FWD", "RWD", "AWD", "4WD"], key="input_drivetrain")
                    if 'fuel_efficiency' in model_features:
                        input_data['fuel_efficiency'] = st.number_input("Fuel Efficiency (km/L)", min_value=0.0,
                                                                        value=10.0, key="input_fuel_efficiency")
                    if 'cylinders' in model_features:
                        input_data['cylinders'] = st.number_input("Number of Cylinders", min_value=0, value=4,
                                                                  key="input_cylinders")

            with tab3:
                st.subheader("Condition & History")
                col1, col2 = st.columns(2)
                with col1:
                    if 'previous_owners' in model_features:
                        input_data['previous_owners'] = st.number_input("Number of Previous Owners", min_value=0,
                                                                        value=1, key="input_previous_owners")
                    if 'accident_history' in model_features:
                        input_data['accident_history'] = st.checkbox("Has Accident History",
                                                                     key="input_accident_history")
                    if 'condition_rating' in model_features:
                        input_data['condition_rating'] = st.selectbox("Condition Rating",
                                                                      ["Excellent", "Good", "Fair", "Poor"],
                                                                      key="input_condition_rating")
                with col2:
                    if 'odometer' in model_features:
                        input_data['odometer'] = st.number_input("Odometer Reading (km)", min_value=0, value=50000,
                                                                 key="input_odometer")
                    if 'warranty_status' in model_features:
                        input_data['warranty_status'] = st.checkbox("Under Warranty", key="input_warranty_status")
                    if 'service_records' in model_features:
                        input_data['service_records'] = st.selectbox("Service Records",
                                                                     ["Complete", "Partial", "None"],
                                                                     key="input_service_records")

            with tab4:
                st.subheader("Market Factors")
                col1, col2 = st.columns(2)
                with col1:
                    if 'location' in model_features:
                        input_data['location'] = st.text_input("Location", key="input_location")
                    if 'season' in model_features:
                        input_data['season'] = st.selectbox("Season",
                                                            ["Spring", "Summer", "Fall", "Winter"], key="input_season")
                with col2:
                    if 'market_demand' in model_features:
                        input_data['market_demand'] = st.slider("Market Demand (1-10)", 1, 10, 5,
                                                                key="input_market_demand")
                    if 'fuel_prices' in model_features:
                        input_data['fuel_prices'] = st.number_input("Current Fuel Price (per L)", min_value=0.0,
                                                                    value=1.0, key="input_fuel_prices")

            with tab5:
                st.subheader("Optional Features")
                col1, col2 = st.columns(2)
                with col1:
                    if 'sunroof' in model_features:
                        input_data['sunroof'] = st.checkbox("Sunroof", key="input_sunroof")
                    if 'navigation' in model_features:
                        input_data['navigation'] = st.checkbox("Navigation System", key="input_navigation")
                    if 'leather_seats' in model_features:
                        input_data['leather_seats'] = st.checkbox("Leather Seats", key="input_leather_seats")
                with col2:
                    if 'parking_sensors' in model_features:
                        input_data['parking_sensors'] = st.checkbox("Parking Sensors", key="input_parking_sensors")
                    if 'backup_camera' in model_features:
                        input_data['backup_camera'] = st.checkbox("Backup Camera", key="input_backup_camera")
                    if 'safety_features' in model_features:
                        input_data['safety_features'] = st.multiselect("Safety Features",
                                                                       ["ABS", "ESP", "Airbags", "Lane Assist",
                                                                        "Blind Spot Detection"],
                                                                       key="input_safety_features")

            with tab6:
                st.subheader("Listing Information")
                col1, col2 = st.columns(2)
                with col1:
                    if 'listing_platform' in model_features:
                        input_data['listing_platform'] = st.selectbox("Listing Platform",
                                                                      ["Dealer Website", "Online Marketplace",
                                                                       "Classified Ads"], key="input_listing_platform")
                    if 'days_listed' in model_features:
                        input_data['days_listed'] = st.number_input("Days Listed", min_value=0, value=7,
                                                                    key="input_days_listed")
                with col2:
                    if 'views' in model_features:
                        input_data['views'] = st.number_input("Number of Views", min_value=0, value=100,
                                                              key="input_views")
                    if 'inquiries' in model_features:
                        input_data['inquiries'] = st.number_input("Number of Inquiries", min_value=0, value=10,
                                                                  key="input_inquiries")
                    if 'seller_type' in model_features:
                        input_data['seller_type'] = st.selectbox("Seller Type",
                                                                 ["Dealer", "Private", "Certified Dealer"],
                                                                 key="input_seller_type")

            if st.button("Predict Price", key="predict_price_manual"):
                try:
                    # Convert multiselect to string for safety_features
                    if 'safety_features' in input_data:
                        input_data['safety_features'] = ', '.join(input_data['safety_features'])

                    input_df = pd.DataFrame([input_data])

                    # --- Pass the original input_df to PDF generation ---
                    original_input_df = input_df.copy()  # Create a copy to preserve original values
                    # --- End of change ---

                    # Ensure all model features are present
                    for col in model_features:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[model_features]

                    processed_df, _ = preprocess_input(input_df, existing_label_encoders=label_encoders)
                    prediction = model.predict(processed_df)
                    processed_df['Predicted_Price'] = prediction
                    decoded_df = decode_labels(processed_df.copy(), label_encoders)
                    log_prediction(decoded_df)

                    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

                    # Show detailed explanation
                    st.subheader("🔍 Why this price?")
                    explanation = explain_prediction(decoded_df.iloc[0], decoded_df.columns)
                    st.write(explanation)

                    # Show feature importance
                    show_feature_importance(model, model_features)

                    # Download options
                    st.download_button("📥 Download CSV", decoded_df.to_csv(index=False), "prediction.csv", "text/csv")
                    # Pass the original_input_df to the PDF generation function
                    pdf_data = generate_enhanced_pdf(decoded_df, model, model_features, original_input_df)
                    st.download_button("📄 Download PDF Report", pdf_data, "prediction_report.pdf", "application/pdf")

                    with st.expander("📧 Email Results"):
                        email = st.text_input("Recipient Email")
                        if st.button("Send Email", key="send_email_manual"):
                            try:
                                send_prediction_email(email, decoded_df, pdf_binary=pdf_data)
                                st.success("Email sent successfully!")
                            except Exception as e:
                                st.error(f"Email error: {e}")
                                st.text(traceback.format_exc())
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.text(traceback.format_exc())
        else:
            st.warning("Please load a model first.")

    elif choice == "Prediction History":
        st.subheader("📜 Prediction History")
        try:
            history_df = get_prediction_history()
            st.write(history_df)
            st.download_button("Download History CSV", history_df.to_csv(index=False), "history.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not load history: {e}")

    elif choice == "Profile":
        profile_page()

    elif choice == "Model Management":
        model_management_page()

    elif choice == "Admin Dashboard":
        admin_dashboard_page()

    elif choice == "Feedback & Support":
        feedback_page()


def admin_dashboard_page():
    st.header("👑 Admin Dashboard")
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access the Admin Dashboard.")
        return
    username = st.session_state.get('username')
    if not check_admin_status(username):
        st.error("You do not have admin privileges.")
        return

    st.subheader("Registered Users")
    # Fetch all users (excluding password hashes and salts)
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, verified, is_admin FROM users")
    users_data = cursor.fetchall()
    conn.close()

    if users_data:
        user_df = pd.DataFrame(users_data, columns=['ID', 'Username', 'Email', 'Verified', 'Is Admin'])
        # Convert boolean-like integers to Yes/No for clarity
        user_df['Verified'] = user_df['Verified'].apply(lambda x: 'Yes' if x == 1 else 'No')
        user_df['Is Admin'] = user_df['Is Admin'].apply(lambda x: 'Yes' if x == 1 else 'No')
        st.dataframe(user_df)
    else:
        st.info("No registered users found.")

    st.markdown("---")
    st.subheader("Trained Models (Across All Users)")
    # Fetch all models
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT m.model_name, m.rmse, m.train_date, u.username FROM models m JOIN users u ON m.user_id = u.id ORDER BY m.train_date DESC")
    models_data = cursor.fetchall()
    conn.close()

    if models_data:
        model_df = pd.DataFrame(models_data, columns=['Model Name', 'RMSE', 'Train Date', 'Trained By'])
        st.dataframe(model_df)
    else:
        st.info("No trained models found.")