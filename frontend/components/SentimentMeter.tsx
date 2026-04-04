type SentimentMeterProps = {
  sentiment: "Positive" | "Neutral" | "Negative";
};

export default function SentimentMeter({ sentiment }: SentimentMeterProps) {
  const config =
    sentiment === "Positive"
      ? { emoji: "😊", className: "sentiment positive" }
      : sentiment === "Negative"
        ? { emoji: "😟", className: "sentiment negative" }
        : { emoji: "😐", className: "sentiment neutral" };

  return (
    <div className="sentiment-card">
      <div className={config.className}>
        <span className="emoji">{config.emoji}</span>
        <span className="label">{sentiment}</span>
      </div>
      <p className="footnote">Powered by Groq Llama + VADER ensemble</p>
    </div>
  );
}
