type EntityBadgesProps = {
  title: string;
  values: string[];
  className: string;
};

export default function EntityBadges({ title, values, className }: EntityBadgesProps) {
  return (
    <div className="entity-section">
      <h4>{title}</h4>
      <div className="badge-wrap">
        {values.length === 0 ? (
          <span className="none">None found</span>
        ) : (
          values.map((value, idx) => (
            <span key={`${value}-${idx}`} className={`badge ${className}`}>
              {value}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
