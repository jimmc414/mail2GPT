Below is an **example reference implementation** that demonstrates how the pieces fit together into a single asynchronous pipeline—using **QStash**, **DigitalOcean Spaces**, **GPT-4**, **o1**, and **Resend**. The goal is to show **one** way to orchestrate the end-to-end flow (receive an email payload, store attachments, process with GPT-4, generate an o1-based response, and finally send via Resend).

> **Note**  
> This sample focuses on the *conceptual flow* and key integration points. Production-ready code will likely need additional error handling, logging, concurrency or rate-limiting strategies, environment variable usage for secrets, etc.

---

## 1. `requirements.txt` Example

You might have the following dependencies:

```txt
flask
boto3
python-dotenv
qstash
openai
resend
# ... etc
```

---

## 2. Folder Structure

Below is a possible folder layout; adapt as needed:

```
my_email_automation/
  ├─ app.py           # Main Flask application (or FastAPI, etc.)
  ├─ config.py        # Configuration loading, environment variables, etc.
  ├─ workers.py       # Code that processes the QStash-queued emails
  ├─ gpt_processing.py # GPT-4 related helper functions
  ├─ do_spaces.py     # DigitalOcean Spaces helper functions
  ├─ resend_client.py # Helper for sending emails via Resend
  ├─ ...
  └─ requirements.txt
```

---

## 3. DigitalOcean Spaces Helper (`do_spaces.py`)

```python
import boto3
from botocore.client import Config

import os

# You'd typically load these from environment variables
DO_ACCESS_KEY = os.getenv("DO_ACCESS_KEY")
DO_SECRET_KEY = os.getenv("DO_SECRET_KEY")
DO_BUCKET_NAME = os.getenv("DO_BUCKET_NAME", "your-space-name")
DO_REGION = "nyc3"  # or whichever region
DO_ENDPOINT = f"https://{DO_REGION}.digitaloceanspaces.com"

session = boto3.session.Session()

s3_client = session.client(
    "s3",
    region_name=DO_REGION,
    endpoint_url=DO_ENDPOINT,
    aws_access_key_id=DO_ACCESS_KEY,
    aws_secret_access_key=DO_SECRET_KEY,
)

def upload_to_spaces(file_name: str, file_content: bytes) -> str:
    """
    Uploads a file to DigitalOcean Spaces and returns a direct URL.
    """
    s3_client.put_object(
        Bucket=DO_BUCKET_NAME, 
        Key=file_name, 
        Body=file_content,
        ACL='private'  # or 'public-read' if you want them publicly accessible
    )
    return f"{DO_ENDPOINT}/{DO_BUCKET_NAME}/{file_name}"
```

---

## 4. GPT-4 Processing Helper (`gpt_processing.py`)

```python
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def analyze_attachment(attachment_url: str) -> str:
    """
    Sends the attachment URL to GPT-4 for analysis.
    For PDFs/images, you would first run OCR or text extraction 
    before sending the content to GPT-4.
    """
    # Here, we are just passing the URL, but in practice you'd fetch 
    # or parse the text content and feed that text into GPT-4.
    messages = [
        {"role": "system", "content": "You are an AI that labels, summarizes, and extracts insights."},
        {
            "role": "user",
            "content": f"Analyze the content at {attachment_url} and return a JSON of labels, summary, and key insights."
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response["choices"][0]["message"]["content"]
```

> **Real-World Note**: For images or scanned PDFs, incorporate OCR (e.g., Tesseract or a cloud OCR API) to extract text, then feed the **text** (instead of the URL) to GPT-4.

---

## 5. Resend Helper (`resend_client.py`)

```python
import os
from resend import Resend

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
resend_client = Resend(api_key=RESEND_API_KEY)

def send_response(to_email: str, subject: str, html_body: str) -> None:
    """
    Sends an email via Resend.
    """
    resend_client.emails.send({
        "from": "Your Name <your_email@example.com>",
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    })
```

---

## 6. Main Flask App with Email Reception & QStash Publish (`app.py`)

This is where incoming emails arrive, presumably from an email service webhook. The route receives the email payload and attachments, then enqueues the data via QStash.

```python
from flask import Flask, request, jsonify
import os, base64, uuid

from do_spaces import upload_to_spaces
from workers import process_email_data  # We'll define the QStash consumer in a separate file
from qstash import QStash

app = Flask(__name__)

# Initialize QStash client
QSTASH_TOKEN = os.getenv("QSTASH_TOKEN")
qstash_client = QStash(token=QSTASH_TOKEN)

@app.route("/email-receive", methods=["POST"])
def email_receive():
    """
    Endpoint to receive incoming emails (likely from an email service).
    Expected payload example:
    {
      "from": "...",
      "subject": "...",
      "text": "...",
      "attachments": [
        {
          "filename": "invoice.pdf",
          "content": "<base64-encoded-bytes>",
          "content-type": "application/pdf"
        },
        ...
      ]
    }
    """
    data = request.json
    sender = data.get("from")
    subject = data.get("subject", "")
    body = data.get("text", "")
    attachments = data.get("attachments", [])

    # Validate & upload each attachment to DO Spaces
    attachment_urls = []
    for attachment in attachments:
        file_name = attachment["filename"]
        # The "content" might be base64-encoded from the inbound email service
        file_content = base64.b64decode(attachment["content"])
        # Optionally validate content type, file size, etc.
        # Then upload
        unique_name = f"{uuid.uuid4()}-{file_name}"
        file_url = upload_to_spaces(unique_name, file_content)
        attachment_urls.append(file_url)

    # Now queue the email + attachments for further GPT & o1 processing
    # using QStash. We'll define the processing route separately.
    qstash_client.message.publish_json(
        url="https://<your-domain>/process-email",  # The route that QStash will POST to
        body={
            "sender": sender,
            "subject": subject,
            "body": body,
            "attachments": attachment_urls
        },
        retries=3,  # Retries if the /process-email endpoint is down
        delay="0s",  # No delay
    )
    return jsonify({"message": "Email received and queued"}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

> **Note**: You can also put your QStash publish code in a separate function, but embedding it in the route is fine for demonstration.

---

## 7. QStash Consumer Endpoint (`workers.py`)

Here we define the `/process-email` route that QStash will call with the queued data. This is where GPT-4 labeling/analysis happens, we call *o1* to generate a final response, and we send the result via Resend.

**Important**: You must verify QStash signatures in production or specify a secret signature to ensure the call is actually from QStash.

```python
from flask import Blueprint, request, jsonify
import requests

from gpt_processing import analyze_attachment
from resend_client import send_response

process_bp = Blueprint("process_bp", __name__)

@process_bp.route("/process-email", methods=["POST"])
def process_email_data():
    """
    This is the QStash consumer that processes queued data:
      1. Uses GPT-4 to label/parse attachments
      2. Calls 'o1' (some external service) to generate custom response
      3. Sends final response to original email sender via Resend
    """
    data = request.json

    sender = data["sender"]
    subject = data["subject"]
    body = data["body"]
    attachment_urls = data["attachments"]

    # Step 1: GPT-4 analysis on attachments
    analysis_results = []
    for url in attachment_urls:
        # Example: store each GPT-4 result in a list
        analysis_json = analyze_attachment(url)
        analysis_results.append(analysis_json)

    # Step 2: Call "o1" with email body + GPT analysis
    #   - This is a placeholder; you'd adapt to how "o1" expects data
    #     and what it returns.
    # Suppose "o1" also runs a text model or logic that returns a recommended response
    o1_response = requests.post("https://api.o1.example.com/generate-response", json={
        "email_subject": subject,
        "email_body": body,
        "analysis_results": analysis_results
    })
    if o1_response.status_code != 200:
        # Log or handle error
        return jsonify({"error": "o1 service failed"}), 500

    # Suppose o1 returns: { "subject": "...", "body_html": "..." }
    o1_data = o1_response.json()
    final_subject = o1_data["subject"]
    final_html = o1_data["body_html"]

    # Step 3: Resend final email
    send_response(
        to_email=sender,
        subject=final_subject,
        html_body=final_html
    )

    return jsonify({"status": "success"}), 200
```

You would then **register** `process_bp` on your Flask app:

```python
# In app.py or a separate "run.py", you can do something like:
from workers import process_bp

app.register_blueprint(process_bp, url_prefix="/")
```

---

## 8. Using QStash Verification

**In production**, you should verify QStash’s signature. With the [QStash Python library](https://github.com/upstash/qstash-py), you’d do something like:

```python
from qstash import Receiver
import os

QSTASH_CURRENT_SIGNING_KEY = os.getenv("QSTASH_CURRENT_SIGNING_KEY")
QSTASH_NEXT_SIGNING_KEY = os.getenv("QSTASH_NEXT_SIGNING_KEY")

receiver = Receiver(
    current_signing_key=QSTASH_CURRENT_SIGNING_KEY,
    next_signing_key=QSTASH_NEXT_SIGNING_KEY,
)

@process_bp.route("/process-email", methods=["POST"])
def process_email_data():
    signature = request.headers.get("Upstash-Signature", "")
    raw_body = request.get_data(as_text=True)

    try:
        receiver.verify(
            signature=signature,
            body=raw_body,
            url="https://<your-domain>/process-email"  # must match exactly
        )
    except Exception as e:
        return jsonify({"error": "Unauthorized"}), 401

    # Then proceed with the logic above:
    data = request.json
    ...
```

---

## 9. Deploying & Running

1. **Set environment variables** (API keys, DO Spaces credentials, QStash keys, Resend key, etc.).
2. **Run** your Flask app. Example:

   ```bash
   export FLASK_APP=app.py
   flask run --port=5000
   ```
3. **Expose** your app to the internet (e.g., a public cloud or via a tool like ngrok).
4. **Configure** QStash & your email service provider’s webhook to POST to `https://<your-domain>/email-receive`.

---

### Putting It All Together

- Inbound Email -> your `/email-receive` route -> uploads attachments to DO Spaces -> QStash queue
- QStash -> hits your `/process-email` route -> GPT-4 analysis -> call `o1` -> final Resend email

This architecture separates **real-time request handling** (the inbound email route) from the **long-running AI tasks** (GPT-4 + calls to `o1`) via the QStash queue. That way, the inbound route quickly returns success while the actual processing can happen asynchronously.

---

## Final Thoughts

- **Security**: Make sure to verify QStash signatures in production.
- **Error Handling**: Add robust error handling ( retries, logging ) around calls to external services (o1, GPT-4, DigitalOcean, Resend).
- **File Size Limits**: Check both your email service’s and DO Spaces’ size constraints.  
- **Testing**: Test thoroughly with sample emails (with/without attachments, multiple attachments, etc.).

---

**That’s it!** With the above blueprint, you have a **complete** reference flow integrating:
- **Flask** or similar for HTTP handling
- **DigitalOcean Spaces** for secure attachment storage
- **GPT-4** for advanced text analysis/labeling
- **QStash** to queue and asynchronously process heavy tasks
- **o1** for specialized response generation
- **Resend** to mail the final answer back to the original sender.
