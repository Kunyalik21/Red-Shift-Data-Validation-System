import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List
from dotenv import load_dotenv


def load_email_credentials() -> tuple[str, str]:
    load_dotenv()
    sender = os.getenv('EMAIL_SENDER')
    password = os.getenv('EMAIL_PASSWORD')
    return sender, password


def send_email_with_attachment(smtp_server: str, smtp_port: int, sender: str, password: str,
                               recipients: List[str], subject: str, html_body: str,
                               attachment_path: str | None = None) -> None:
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    if attachment_path and os.path.isfile(attachment_path):
        with open(attachment_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
        # Add Content-ID so the attachment can be referenced via cid: in the email body
        part.add_header('Content-ID', '<report_attachment>')
        msg.attach(part)

    # Sanitize potential spaced app passwords (Gmail displays with spaces)
    if password:
        password = password.replace(' ', '')

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())


