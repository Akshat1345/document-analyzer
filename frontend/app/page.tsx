"use client";

import { useEffect, useMemo, useState } from "react";

import DropZone from "../components/DropZone";
import ResultCard from "../components/ResultCard";

type AnalysisResponse = {
  status: "success";
  fileName: string;
  documentId?: string;
  summary: string;
  entities: {
    names: string[];
    dates: string[];
    organizations: string[];
    amounts: string[];
    emails: string[];
    phones: string[];
  };
  sentiment: "Positive" | "Neutral" | "Negative";
};

type QaResponse = {
  status: "success";
  documentId: string;
  question: string;
  answer: string;
  citations: string[];
};

const PROCESS_STEPS = [
  "📄 Extracting text...",
  "🔍 Running entity recognition...",
  "🧠 Analyzing sentiment...",
  "✍️ Generating summary...",
];

function detectFileType(fileName: string): "pdf" | "docx" | "image" {
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".pdf")) return "pdf";
  if (lower.endsWith(".docx")) return "docx";
  return "image";
}

function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] ?? "");
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingStep, setProcessingStep] = useState("");
  const [copied, setCopied] = useState(false);
  const [question, setQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaResult, setQaResult] = useState<QaResponse | null>(null);
  const [qaError, setQaError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading) return;
    let index = 0;
    setProcessingStep(PROCESS_STEPS[index]);
    const id = setInterval(() => {
      index = (index + 1) % PROCESS_STEPS.length;
      setProcessingStep(PROCESS_STEPS[index]);
    }, 800);
    return () => clearInterval(id);
  }, [loading]);

  const canAnalyze = useMemo(() => Boolean(file) && !loading, [file, loading]);

  const handleFileSelect = (selectedFile: File | null) => {
    setFile(selectedFile);
    setError(null);
    setResult(null);
    setCopied(false);
    setQuestion("");
    setQaResult(null);
    setQaError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const payload = {
        fileName: file.name,
        fileType: detectFileType(file.name),
        fileBase64: await toBase64(file),
      };

      const response = await fetch(`${apiUrl}/api/document-analyze`, {
        method: "POST",
        headers: {
          "x-api-key": process.env.NEXT_PUBLIC_API_KEY || "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok || data.status === "error") {
        throw new Error(data.message || "Analysis failed");
      }

      setResult(data as AnalysisResponse);
      setQaResult(null);
      setQaError(null);
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
      setProcessingStep("");
    }
  };

  const reset = () => {
    setFile(null);
    setLoading(false);
    setResult(null);
    setError(null);
    setProcessingStep("");
    setQuestion("");
    setQaResult(null);
    setQaError(null);
  };

  const handleAskQuestion = async () => {
    if (!result?.documentId || !question.trim()) return;

    setQaLoading(true);
    setQaError(null);
    setQaResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/document-qa`, {
        method: "POST",
        headers: {
          "x-api-key": process.env.NEXT_PUBLIC_API_KEY || "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          documentId: result.documentId,
          question: question.trim(),
          topK: 4,
        }),
      });

      const data = await response.json();
      if (!response.ok || data.status === "error") {
        throw new Error(data.message || "Q&A failed");
      }

      setQaResult(data as QaResponse);
    } catch (err) {
      setQaError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setQaLoading(false);
    }
  };

  const copyJson = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <main className="page">
      <header className="hero">
        <h1>DocuMind AI 🧠</h1>
        <p>Intelligent Document Analysis</p>
      </header>

      <DropZone file={file} onFileSelect={handleFileSelect} />

      <button
        className="analyze-btn"
        disabled={!canAnalyze}
        onClick={handleAnalyze}
      >
        {loading ? "Analyzing..." : "Analyze Document"}
      </button>

      {loading && <div className="processing">{processingStep}</div>}

      {error && (
        <div className="error-banner">
          <p>{error}</p>
          <button onClick={handleAnalyze} disabled={!file || loading}>
            Retry Analyze
          </button>
          <button onClick={reset} disabled={loading}>
            Choose Another File
          </button>
        </div>
      )}

      {result && (
        <>
          <ResultCard result={result} />
          {result.documentId && (
            <section className="card qa-card">
              <h3>Ask Questions About This Document</h3>
              <p className="footnote">Document ID: {result.documentId}</p>
              <div className="qa-row">
                <input
                  className="qa-input"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything from this specific document..."
                />
                <button
                  className="analyze-btn"
                  onClick={handleAskQuestion}
                  disabled={qaLoading || !question.trim()}
                >
                  {qaLoading ? "Asking..." : "Ask"}
                </button>
              </div>

              {qaError && <p className="qa-error">{qaError}</p>}

              {qaResult && (
                <div className="qa-answer">
                  <h4>Answer</h4>
                  <p>{qaResult.answer}</p>
                  {qaResult.citations.length > 0 && (
                    <p className="footnote">Sources: {qaResult.citations.join(", ")}</p>
                  )}
                </div>
              )}
            </section>
          )}
          {!result.documentId && (
            <section className="card qa-card">
              <h3>Ask Questions About This Document</h3>
              <p className="qa-error">
                Document ID missing in response. Please re-run Analyze once to enable document-scoped Q&amp;A.
              </p>
            </section>
          )}
          <div className="json-actions">
            <button onClick={copyJson}>Copy JSON</button>
            {copied && <span>Copied!</span>}
          </div>
        </>
      )}
    </main>
  );
}
