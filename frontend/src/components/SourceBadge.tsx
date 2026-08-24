const LABELS: Record<string, string> = {
  policy_kb: "Policy KB",
  return_risk_tool: "Return-Risk Model",
  image_classifier_tool: "Image Classifier",
};

export default function SourceBadge({ source }: { source: string }) {
  return (
    <span
      className="mono"
      style={{
        background: "var(--surface-2)",
        color: "var(--text-secondary)",
        border: "1px solid var(--border)",
        padding: "2px 8px",
        borderRadius: 6,
        fontSize: 11,
      }}
    >
      {LABELS[source] ?? source}
    </span>
  );
}
