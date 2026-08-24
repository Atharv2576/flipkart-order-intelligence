import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Assistant from "./screens/Assistant";
import Insights from "./screens/Insights";
import OrderDetail from "./screens/OrderDetail";
import Orders from "./screens/Orders";
import Status from "./screens/Status";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Assistant />} />
          <Route path="/orders" element={<Orders />} />
          <Route path="/orders/:orderId" element={<OrderDetail />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/status" element={<Status />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
