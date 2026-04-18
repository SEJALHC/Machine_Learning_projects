import smtplib
from email.mime.text import MIMEText

def send_email(receiver):
    msg = MIMEText("Duplicate files removed successfully.")
    msg["Subject"] = "Report"
    msg["From"] = "your_email@gmail.com"
    msg["To"] = receiver

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("your_email@gmail.com", "your_app_password")

    server.send_message(msg)
    server.quit()