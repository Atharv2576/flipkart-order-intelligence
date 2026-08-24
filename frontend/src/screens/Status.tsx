import { useEffect, useState } from "react";
import { fetchStatus } from "../lib/api";
import type { StatusResponse } from "../lib/types";

export default function Status() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus()
      .then(setStatus)
      .catch((err) => setError(err instanceof Error ? err.message : "Backend unreachable."));
  }, []);

  return (
    <div style={{ padding: "24px 32px", maxWidth: 560 }}>
      <h1 style={{ fontSize: 18, marginBottom: 4 }}>System Status</h1>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 0 }}>
        Live readiness of every component the agent depends on.
      </p>

      {error && (
        <div
          style={{
            marginTop: 16,
            padding: "10px 14px",
            borderRadius: 8,
            background: "var(--status-critical-soft)",
            color: "var(--status-critical)",
            fontSize: 13,
          }}
        >
          {error} -- is the backend running (`python3 -m api.main`)?
        </div>
      )}

      {status && (
        <>
          <div
            style={{
              margin: "16px 0",
              padding: "10px 14px",
              borderRadius: 8,
              background: "var(--surface-2)",
              fontSize: 13,
            }}
          >
            Mode: <strong className="mono">{status.mode}</strong> &middot; groundedness threshold{" "}
            <strong className="mono">{status.groundedness_threshold}</strong> &middot; intent floor{" "}
            <strong className="mono">{status.intent_routing_floor}</strong>
          </div>

          {Object.entries(status.components).map(([name, comp]) => (
            <div
              key={name}
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "10px 0",
                borderBottom: "1px solid var(--border)",
                fontSize: 14,
              }}
            >
              <span>{name.replace(/_/g, " ")}</span>
              <span
                style={{
                  color: comp.ready ? "var(--status-good)" : "var(--status-critical)",
                  fontWeight: 600,
                }}
              >
                {comp.ready ? "READY" : "NOT READY"}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
