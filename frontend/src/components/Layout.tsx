import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Assistant", end: true },
  { to: "/orders", label: "Orders" },
  { to: "/insights", label: "Insights" },
  { to: "/status", label: "System" },
];

export default function Layout() {
  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Primary">
        <div style={{ padding: "0 8px 20px", fontWeight: 700, fontSize: 15, flexShrink: 0 }}>
          Flipkart Order
          <br />
          Intelligence
        </div>
        <div className="app-nav-links">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              style={({ isActive }) => ({
                display: "block",
                padding: "9px 10px",
                borderRadius: 8,
                marginBottom: 2,
                fontSize: 14,
                fontWeight: 500,
                textDecoration: "none",
                color: isActive ? "var(--accent)" : "var(--text-secondary)",
                background: isActive ? "var(--accent-soft)" : "transparent",
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
