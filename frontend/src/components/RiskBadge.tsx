const STYLES: Record<string, { bg: string; fg: string }> = {
  Low: { bg: "var(--status-good-soft)", fg: "var(--status-good)" },
  Medium: { bg: "var(--status-warning-soft)", fg: "var(--status-warning)" },
  High: { bg: "var(--status-critical-soft)", fg: "var(--status-critical)" },
};

export default function RiskBadge({ bucket }: { bucket: string }) {
  const style = STYLES[bucket] ?? STYLES.Medium;
  return (
    <span
      style={{
        background: style.bg,
        color: style.fg,
        padding: "2px 10px",
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: 0.2,
      }}
    >
      {bucket.toUpperCase()} RISK
    </span>
  );
}
