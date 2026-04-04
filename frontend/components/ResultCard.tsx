import EntityBadges from "./EntityBadges";
import SentimentMeter from "./SentimentMeter";

type AnalysisResponse = {
  status: "success";
  fileName: string;
  summary: string;
  entities: {
    names: string[];
    dates: string[];
    organizations: string[];
    amounts: string[];
  };
  sentiment: "Positive" | "Neutral" | "Negative";
};

export default function ResultCard({ result }: { result: AnalysisResponse }) {
  return (
    <div className="result-grid">
      <section className="card">
        <h3>Summary</h3>
        <p>{result.summary}</p>
        <span className="word-count">
          {result.summary.split(/\s+/).filter(Boolean).length} words
        </span>
      </section>

      <section className="card">
        <h3>Entities</h3>
        <EntityBadges
          title="Names"
          values={result.entities.names}
          className="name"
        />
        <EntityBadges
          title="Dates"
          values={result.entities.dates}
          className="date"
        />
        <EntityBadges
          title="Organizations"
          values={result.entities.organizations}
          className="org"
        />
        <EntityBadges
          title="Amounts"
          values={result.entities.amounts}
          className="amount"
        />
      </section>

      <section className="card">
        <h3>Sentiment</h3>
        <SentimentMeter sentiment={result.sentiment} />
      </section>
    </div>
  );
}
