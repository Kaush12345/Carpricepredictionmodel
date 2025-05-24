import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_prediction_email(to_email, prediction_df, pdf_path=None, pdf_binary=None):
    msg = EmailMessage()
    msg["Subject"] = "Your Used Car Price Prediction Report"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email

    body = "Hi,\n\nAttached is your used car price prediction report.\n\nPrediction Summary:\n"
    for col in prediction_df.columns:
        body += f"{col}: {prediction_df[col].values[0]}\n"

    msg.set_content(body)

    # Attach PDF
    if pdf_binary:
        msg.add_attachment(pdf_binary, maintype="application", subtype="pdf", filename="prediction.pdf")
    elif pdf_path:
        with open(pdf_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="prediction.pdf")

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("✅ Email sent successfully.")
