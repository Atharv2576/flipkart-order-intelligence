export interface RetrievedDocument {
  chunk_id: string;
  document_id: string;
  document_title: string;
  chunk_index: number;
  chunk_text: string;
  score: number;
}

export interface Groundedness {
  grounded: boolean;
  best_score: number;
  threshold: number;
}

export interface ToolResult {
  [key: string]: unknown;
}

export interface ChatResponse {
  conversation_id: string;
  answer: string;
  source: "policy_kb" | "return_risk_tool" | "image_classifier_tool";
  confidence: number;
  intent: string | null;
  node_path: string[];
  order_id: number | null;
  blocked: boolean;
  groundedness: Groundedness | null;
  retrieved_documents: RetrievedDocument[] | null;
  tool_result: ToolResult | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
}

export interface OrderFeatures {
  product_category: string;
  price_inr: number;
  discount_pct: number;
  payment_method: string;
  customer_tenure_days: number;
  num_previous_orders: number;
  num_previous_returns: number;
  delivery_distance_km: number;
  delivery_days: number;
  is_weekend_order: number;
  rating_given: number | null;
}

export interface Order extends OrderFeatures {
  order_id: number;
  returned: number;
}

export interface RiskResult {
  return_probability: number;
  risk_bucket: "Low" | "Medium" | "High";
  threshold_rf: number;
}

export interface OrderDetail {
  order_id: number;
  order_features: OrderFeatures;
  risk: RiskResult;
}

export interface StatusResponse {
  components: Record<string, { ready: boolean; detail?: string }>;
  mode: "mock_llm" | "live_llm";
  groundedness_threshold: number;
  intent_routing_floor: number;
}

export interface PolicyDocument {
  id: string;
  title: string;
  text: string;
}
