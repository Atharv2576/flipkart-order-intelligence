import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchOrders } from "../lib/api";
import type { Order } from "../lib/types";

const CATEGORIES = ["", "Apparel", "Electronics", "Home", "Footwear", "Beauty"];
const PAYMENT_METHODS = ["", "COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"];

export default function Orders() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [total, setTotal] = useState(0);
  const [category, setCategory] = useState("");
  const [payment, setPayment] = useState("");
  const [orderIdSearch, setOrderIdSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const limit = 25;

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchOrders({
      product_category: category || undefined,
      payment_method: payment || undefined,
      order_id: orderIdSearch ? Number(orderIdSearch) : undefined,
      limit,
      offset,
    })
      .then((res) => {
        setOrders(res.orders);
        setTotal(res.total);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load orders."))
      .finally(() => setLoading(false));
  }, [category, payment, orderIdSearch, offset]);

  return (
    <div style={{ padding: "24px 32px" }}>
      <h1 style={{ fontSize: 18, marginBottom: 4 }}>Order Explorer</h1>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 0 }}>
        Real rows from orders_dataset.csv, {total.toLocaleString()} orders total.
      </p>

      <div style={{ display: "flex", gap: 10, margin: "16px 0" }}>
        <input
          placeholder="Search by order id…"
          value={orderIdSearch}
          onChange={(e) => {
            setOrderIdSearch(e.target.value);
            setOffset(0);
          }}
          style={inputStyle}
        />
        <select
          value={category}
          onChange={(e) => {
            setCategory(e.target.value);
            setOffset(0);
          }}
          style={inputStyle}
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c || "All categories"}
            </option>
          ))}
        </select>
        <select
          value={payment}
          onChange={(e) => {
            setPayment(e.target.value);
            setOffset(0);
          }}
          style={inputStyle}
        >
          {PAYMENT_METHODS.map((p) => (
            <option key={p} value={p}>
              {p || "All payment methods"}
            </option>
          ))}
        </select>
      </div>

      {error && <div style={{ color: "var(--status-critical)" }}>{error}</div>}
      {loading && <div style={{ color: "var(--text-muted)" }}>Loading…</div>}

      {!loading && orders.length === 0 && !error && (
        <div style={{ color: "var(--text-muted)", padding: 24 }}>No orders match these filters.</div>
      )}

      {orders.length > 0 && (
        <div style={{ overflow: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ background: "var(--surface-2)", textAlign: "left" }}>
                {["Order", "Category", "Price (₹)", "Payment", "Delivery days", "Returned"].map((h) => (
                  <th key={h} style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.order_id} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px 12px" }}>
                    <Link to={`/orders/${o.order_id}`} style={{ color: "var(--accent)", textDecoration: "none" }}>
                      #{o.order_id}
                    </Link>
                  </td>
                  <td style={{ padding: "8px 12px" }}>{o.product_category}</td>
                  <td className="mono" style={{ padding: "8px 12px" }}>
                    {o.price_inr.toLocaleString()}
                  </td>
                  <td style={{ padding: "8px 12px" }}>{o.payment_method}</td>
                  <td className="mono" style={{ padding: "8px 12px" }}>
                    {o.delivery_days}
                  </td>
                  <td style={{ padding: "8px 12px" }}>{o.returned ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginTop: 14, alignItems: "center" }}>
        <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))} style={pageBtn}>
          Prev
        </button>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
          {offset + 1}-{Math.min(offset + limit, total)} of {total}
        </span>
        <button disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)} style={pageBtn}>
          Next
        </button>
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderRadius: 8,
  border: "1px solid var(--border)",
  background: "var(--surface-1)",
  color: "var(--text-primary)",
  fontSize: 13,
};

const pageBtn: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: 8,
  border: "1px solid var(--border)",
  background: "var(--surface-1)",
  cursor: "pointer",
  fontSize: 13,
};
