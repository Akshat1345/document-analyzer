from pathlib import Path

from app.extractors.pdf_extractor import PDFExtractor
from app.extractors.docx_extractor import DOCXExtractor

for name, extractor in [
    ("sample1-Technology Industry Analysis.pdf", PDFExtractor()),
    ("sample2-Cybersecurity Incident Report.docx", DOCXExtractor()),
]:
    raw = (Path('test') / name).read_bytes()
    text, meta = extractor.extract(raw)
    print('\n###', name)
    print(meta)
    print(text[:4000])
    print('---')
