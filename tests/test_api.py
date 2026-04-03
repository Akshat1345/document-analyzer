"""API integration tests for contract compliance and key scenarios."""


def test_missing_api_key(test_client, sample_pdf_b64) -> None:
    """POST without x-api-key header should return 401."""

    payload = {"fileName": "sample1.pdf", "fileType": "pdf", "fileBase64": sample_pdf_b64}
    response = test_client.post("/api/document-analyze", json=payload)
    assert response.status_code == 401


def test_invalid_api_key(test_client, sample_pdf_b64, invalid_headers) -> None:
    """POST with wrong x-api-key header should return 401."""

    payload = {"fileName": "sample1.pdf", "fileType": "pdf", "fileBase64": sample_pdf_b64}
    response = test_client.post("/api/document-analyze", json=payload, headers=invalid_headers)
    assert response.status_code == 401


def test_response_structure(test_client, sample_pdf_b64, valid_headers) -> None:
    """Successful response should match strict required shape."""

    payload = {"fileName": "sample1.pdf", "fileType": "pdf", "fileBase64": sample_pdf_b64}
    response = test_client.post("/api/document-analyze", json=payload, headers=valid_headers)
    data = response.json()
    assert response.status_code == 200
    assert set(data.keys()) == {"status", "fileName", "documentId", "summary", "entities", "sentiment"}
    assert set(data["entities"].keys()) == {
        "names",
        "dates",
        "organizations",
        "amounts",
        "emails",
        "phones",
    }
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]


def test_invalid_base64(test_client, valid_headers) -> None:
    """Invalid base64 should return status=error and not crash."""

    payload = {"fileName": "bad.pdf", "fileType": "pdf", "fileBase64": "not_valid_base64!!!"}
    response = test_client.post("/api/document-analyze", json=payload, headers=valid_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_health_endpoint(test_client) -> None:
    """Health endpoint should return 200 and healthy status."""

    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_pdf_processing(test_client, sample_pdf_b64, valid_headers) -> None:
    """Technology PDF should return positive sentiment and known organizations."""

    payload = {
        "fileName": "sample1-Technology_Industry_Analysis.pdf",
        "fileType": "pdf",
        "fileBase64": sample_pdf_b64,
    }
    response = test_client.post("/api/document-analyze", json=payload, headers=valid_headers)
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert any(org in data["entities"]["organizations"] for org in ["Google", "Microsoft"])


def test_docx_processing(test_client, sample_docx_b64, valid_headers) -> None:
    """Cybersecurity DOCX should return negative sentiment."""

    payload = {
        "fileName": "sample2-Cybersecurity_Incident_Report.docx",
        "fileType": "docx",
        "fileBase64": sample_docx_b64,
    }
    response = test_client.post("/api/document-analyze", json=payload, headers=valid_headers)
    assert response.json()["sentiment"] == "Negative"
