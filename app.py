
from flask import Flask, request, jsonify
import os, base64, uuid

from do_spaces import upload_to_spaces
from workers import process_email_data
from qstash import QStash

app = Flask(__name__)

QSTASH_TOKEN = os.getenv("QSTASH_TOKEN")
qstash_client = QStash(token=QSTASH_TOKEN)

@app.route("/email-receive", methods=["POST"])
def email_receive():
    data = request.json
    sender = data.get("from")
    subject = data.get("subject", "")
    body = data.get("text", "")
    attachments = data.get("attachments", [])

    attachment_urls = []
    for attachment in attachments:
        file_name = attachment["filename"]
        file_content = base64.b64decode(attachment["content"])
        unique_name = f"{uuid.uuid4()}-{file_name}"
        file_url = upload_to_spaces(unique_name, file_content)
        attachment_urls.append(file_url)

    qstash_client.message.publish_json(
        url="https://<your-domain>/process-email",
        body={
            "sender": sender,
            "subject": subject,
            "body": body,
            "attachments": attachment_urls
        },
        retries=3,
        delay="0s",
    )
    return jsonify({"message": "Email received and queued"}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
