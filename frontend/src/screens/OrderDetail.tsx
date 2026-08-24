import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import RiskBadge from "../components/RiskBadge";
import { fetchOrderDetail } from "../lib/api";
import type { OrderDetail as OrderDetailType } from "../lib/types";

export default function OrderDetail() {
  const { orderId } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<OrderDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!orderId) return;
    fetchOrderDetail(Number(orderId))
      .then(setDetail)
      .catch((err) => setError(err instanceof Error ? err.message : "Order not found."));
  }, [orderId]);

  if (error) {
    return (
      <div style={{ padding: 32 }}>
        <Link to="/orders">&larr; back to orders</Link>
        <div style={{ color: "var(--status-critical)", marginTop: 16 }}>{error}</div>
      </div>
    );
  }
  if (!detail) return <div style={{ padding: 32, color: "var(--text-muted)" }}>Loading…</div>;

  const f = detail.order_features;

  return (
    <div style={{ padding: "24px 32px", maxWidth: 720 }}>
      <Link to="/orders" style={{ fontSize: 13, color: "var(--text-muted)" }}>
        &larr; back to orders
      </Link>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "10px 0 20px" }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>Order #{detail.order_id}</h1>
        <RiskBadge bucket={detail.risk.risk_bucket} />
      </div>

      <div
        style={{
          background: "var(--surface-1)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          padding: 20,
          marginBottom: 20,
        }}
      >
        <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 6 }}>
          Predicted return probability
        </div>
        <div className="mono" style={{ fontSize: 28, fontWeight: 700 }}>
          {(detail.risk.return_probability * 100).toFixed(2)}%
        </div>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
          from the tuned Random Forest, anchored to t*_rf = {detail.risk.threshold_rf}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
        {[
          ["Category", f.product_category],
          ["Payment method", f.payment_method],
          ["Price", `₹${f.price_inr.toLocaleString()}`],
          ["Discount", `${f.discount_pct}%`],
          ["Customer tenure", `${f.customer_tenure_days} days`],
          ["Previous orders", f.num_previous_orders],
          ["Previous returns", f.num_previous_returns],
          ["Delivery distance", `${f.delivery_distance_km} km`],
          ["Delivery days", f.delivery_days],
          ["Weekend order", f.is_weekend_order ? "Yes" : "No"],
          ["Rating given", f.rating_given ?? "missing"],
        ].map(([label, value]) => (
          <div key={label as string} style={{ fontSize: 13 }}>
            <div style={{ color: "var(--text-muted)" }}>{label}</div>
            <div className="mono">{value}</div>
          </div>
        ))}
      </div>

      <button
        onClick={() => navigate(`/?ask=${encodeURIComponent(`Score the return risk for order ${detail.order_id}.`)}`)}
        style={{
          padding: "10px 18px",
          borderRadius: 10,
          border: "none",
          background: "var(--accent)",
          color: "white",
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        Ask the AI about this order
      </button>
    </div>
  );
}
