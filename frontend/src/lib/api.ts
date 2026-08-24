import type {
  ChatResponse,
  Order,
  OrderDetail,
  OrderFeatures,
  PolicyDocument,
  RiskResult,
  StatusResponse,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || `Request failed (${response.status})`, response.status);
  }
  return response.json();
}

export function sendChatMessage(
  message: string,
  conversationId: string | null,
  imagePath?: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId, image_path: imagePath }),
  });
}

export function resetConversation(): Promise<{ conversation_id: string }> {
  return request("/api/conversations/reset", { method: "POST" });
}

export function fetchOrders(params: {
  order_id?: number;
  product_category?: string;
  payment_method?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; orders: Order[] }> {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") search.set(key, String(value));
  });
  return request(`/api/orders?${search.toString()}`);
}

export function fetchOrderDetail(orderId: number): Promise<OrderDetail> {
  return request(`/api/orders/${orderId}`);
}

export function scoreReturnRisk(orderFeatures: OrderFeatures): Promise<RiskResult> {
  return request("/api/return-risk", {
    method: "POST",
    body: JSON.stringify({ order_features: orderFeatures }),
  });
}

export function fetchPolicies(): Promise<{ count: number; documents: PolicyDocument[] }> {
  return request("/api/policies");
}

export function fetchSampleImages(): Promise<
  { filename: string; class: string; class_index: number; test_split_index: number }[]
> {
  return request("/api/sample-images");
}

export function classifySampleImage(filename: string): Promise<{
  predicted_class: string;
  confidence: number;
  top3: { class: string; confidence: number }[];
}> {
  return request(`/api/classify?filename=${encodeURIComponent(filename)}`, { method: "POST" });
}

export function fetchStatus(): Promise<StatusResponse> {
  return request("/api/status");
}

export { API_URL, ApiError };
