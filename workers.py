
from flask import Blueprint, request, jsonify
import requests

from gpt_processing import analyze_attachment
from resend_client import send_response

process_bp = Blueprint("process_bp", __name__)

@process_bp.route("/process-email", methods=["POST"])
def process_email_data():
    data = request.json

    sender = data["sender"]
    subject = data["subject"]
    body = data["body"]
    attachment_urls = data["attachments"]

    analysis_results = []
    for url in attachment_urls:
        analysis_json = analyze_attachment(url)
        analysis_results.append(analysis_json)

    o1_response = requests.post("https://api.o1.example.com/generate-response", json={
        "email_subject": subject,
        "email_body": body,
        "analysis_results": analysis_results
    })
    if o1_response.status_code != 200:
        return jsonify({"error": "o1 service failed"}), 500

    o1_data = o1_response.json()
    final_subject = o1_data["subject"]
    final_html = o1_data["body_html"]

    send_response(
        to_email=sender,
        subject=final_subject,
        html_body=final_html
    )

    return jsonify({"status": "success"}), 200
