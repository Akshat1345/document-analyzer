type EntityBadgesProps = {
  title: string;
  values?: string[];
  className: string;
};

export default function EntityBadges({
  title,
  values,
  className,
}: EntityBadgesProps) {
  const safeValues = Array.isArray(values) ? values : [];

  return (
    <div className="entity-section">
      <h4>{title}</h4>
      <div className="badge-wrap">
        {safeValues.length === 0 ? (
          <span className="none">None found</span>
        ) : (
          safeValues.map((value, idx) => (
            <span key={`${value}-${idx}`} className={`badge ${className}`}>
              {value}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
