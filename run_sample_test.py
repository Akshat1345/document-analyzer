from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx


def get_api_key() -> str:
    env_path = Path('.env')
    for line in env_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('API_KEY='):
            return line.split('=', 1)[1].strip()
    raise RuntimeError('API_KEY not found in .env')


def main() -> None:
    api_key = get_api_key()
    test_dir = Path('test')
    files = sorted(p for p in test_dir.iterdir() if p.is_file())

    ext_to_type = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
    }

    results = []
    for file_path in files:
        file_type = ext_to_type.get(file_path.suffix.lower())
        if not file_type:
            results.append(
                {
                    'file': file_path.name,
                    'status': 'skipped',
                    'reason': f'unsupported extension: {file_path.suffix}',
                }
            )
            continue

        payload = {
            'fileName': file_path.name,
            'fileType': file_type,
            'fileBase64': base64.b64encode(file_path.read_bytes()).decode('utf-8'),
        }

        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    'http://localhost:8000/api/document-analyze',
                    headers={'x-api-key': api_key, 'Content-Type': 'application/json'},
                    json=payload,
                )

            try:
                body = response.json()
            except Exception:
                body = {'raw': response.text[:400]}

            results.append(
                {
                    'file': file_path.name,
                    'http_status': response.status_code,
                    'body': body,
                }
            )
        except Exception as exc:
            results.append(
                {
                    'file': file_path.name,
                    'http_status': None,
                    'body': {'status': 'request_failed', 'message': str(exc)},
                }
            )

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
