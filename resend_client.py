
import os
from resend import Resend

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
resend_client = Resend(api_key=RESEND_API_KEY)

def send_response(to_email: str, subject: str, html_body: str) -> None:
    resend_client.emails.send({
        "from": "Your Name <your_email@example.com>",
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    })
