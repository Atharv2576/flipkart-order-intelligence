# AI agent upgrade -- final report

## 1. What "predefined-question chatbot" actually meant here

Before this upgrade, the agent already used embedding-based nearest-few-shot-exemplar
intent routing (`part3/intent.py`) and genuine FAISS retrieval (`part3/retrieval.py`) --
not a literal `if question == X` table. Live-testing the graph at the start of this
upgrade found the real problems were narrower but still genuine:

- `"hi"` had no close exemplar at all, fell below the routing floor, defaulted to
  `policy`, found nothing in the KB, and returned a groundedness refusal that read
  exactly like a denial of service.
- `"what can you help me with?"` landed nearest to a `product_category` exemplar and
  produced *"I need an image path to classify a product photo."*
- `"Compare footwear and electronics return policies"` only ever returned one chunk of
  text; the "related policies" list was titles only, no content.
- A multi-part question (damaged + COD + footwear in one message) only answered the
  generic footwear window and ignored the rest.
- `return_risk` required a known `order_id`; there was no slot-filling from free text
  and no explanation of *why* an order scored the way it did.
- There was no product-level catalog -- only 16 category-level policy documents.

Two additional bugs were found and fixed *during* this upgrade, both caught by actually
running the new code rather than reasoning about it:

- Adding a `conversational` intent bucket initially miscl assified "explain the return
  policy" / "summarize all policies" style questions as `return_risk`, because no
  existing policy exemplar used that general phrasing and it drifted lexically toward
  "return risk"/"gets returned". Fixed by adding two policy exemplars covering that
  phrasing (`part3/intent.py`).
- Naive clause-splitting on "and" turned one coherent question ("what happens if I open
  a pair of headphones and then decide I don't want them?") into two weak, noisy
  half-questions that accidentally landed on two different low-confidence documents and
  falsely triggered multi-clause mode. Fixed with a per-clause confidence floor
  (`MULTI_CLAUSE_CONFIDENCE_FLOOR`, `part3/config.py`) that only lets a clause count
  toward "these are genuinely different topics" if its own top match is at least
  plausible.
- A user answering a pending return-risk clarification ("It took 6 days to arrive.")
  was being reclassified as a fresh, unrelated policy question each turn, since that
  reply doesn't resemble any `return_risk` exemplar on its own. Fixed by tracking
  whether the previous turn ended in a clarification and, if this turn's message
  contributed a genuinely new order-relevant field, treating it as a continuation
  (`part3/graph.py::intent_node`).

## 2. Architecture

```
START -> guard_node -> intent_node -> [route_by_intent]
                                        |-- blocked          -> response_node
                                        |-- conversational   -> response_node   (NEW)
                                        |-- policy           -> retrieval_node  -> response_node
                                        |-- return_risk      -> tool_node       -> response_node
                                        |-- product_category -> tool_node       -> response_node
```

One new route was added (`conversational`); the required five-node, one-conditional-edge
shape is unchanged, so every original graph-routing test still passes unmodified.
Everything else -- comparison, multi-part questions, product-catalog lookups,
slot-filling, risk explanations -- is a *behavior* layered on the existing four routes,
matching the brief's own instruction to map additional conversational intents onto the
required tool/RAG pathways rather than inventing new required routes.

### Intent detection
Still nearest-few-shot-exemplar in embedding space (`part3/intent.py`), now with a fifth
bucket (`conversational`: greeting/general_help/thanks/farewell) and two independent
floors (`INTENT_ROUTING_FLOOR`, `CONVERSATIONAL_ROUTING_FLOOR`) so a message can resolve
to a genuine required intent, a genuine conversational intent, or the pre-existing
below-floor `policy` fallback -- without changing that fallback's behavior for the
one existing test that depends on it.

### RAG / retrieval
`part3/retrieval.py::retrieve_for_message` splits a message into clauses
(`part3/query_decomposition.py`, generic English coordination -- commas/and/but/"compare
X and Y" -- never a domain word list). Two or more clauses that each independently
ground on a *different* document (past `MULTI_CLAUSE_CONFIDENCE_FLOOR`) trigger
`compose_comparison_answer`, which answers each clause with its own citation instead of
one chunk standing in for the whole question. A single-topic message is byte-identical
to the original single-query behavior.

### Product catalog (new knowledge layer)
`scripts/generate_product_catalog.py` deterministically generates 55 synthetic SKUs
(seeded `np.random.default_rng`, same pattern as `generate_orders.py`) across the same
five categories Part 1 already uses, with per-category return windows and exchange
eligibility pinned to the real policy KB (POL01-POL05, POL15) rather than invented
separately. `part3/product_catalog.py` offers both a real structured filter
(`query_products`, for "which products support COD" style enumerable questions) and a
semantic FAISS search (`search_products`, its own index at `data/product_index/`,
entirely separate from the required `data/policy_index/`). Catalog evidence is only
mixed into retrieval when the message itself is asking about product/catalog info
(`is_catalog_relevant`, its own few-shot gate) -- a message describing the user's own
order never gets a random synthetic SKU standing in for the real policy answer.

### Return-risk slot-filling and explanation
`part3/slot_extraction.py` parses price, payment method, product category, and delivery
delay from free text, matching payment method and category against the model's own
closed vocabulary (`COD/Prepaid_Card/Prepaid_UPI/Wallet`, `Apparel/Electronics/Home/
Footwear/Beauty`) -- category resolution uses the same nearest-few-shot-exemplar
mechanism as intent routing, applied to short candidate word spans rather than the
whole noisy sentence. `Conversation.order_features` (`part3/agent.py`) accumulates these
across turns, never overwriting an already-known value. When `return_risk` fires without
enough of the fields that are both extractable and genuinely important
(`REQUIRED_FOR_ESTIMATE` in `part3/tools.py`: category/price/payment/delivery days), the
agent asks only for what's missing. Explanations (`part3/risk_explanation.py`) cite only
features with meaningfully positive permutation importance from Part 1's own held-out
report (payment_method +0.0980 ROC-AUC, price_inr +0.0102, num_previous_returns
+0.0085, product_category +0.0060, delivery_days +0.0026) that are actually present on
the order being explained -- no invented per-instance causal claim.

### MOCK_LLM (unchanged as the required default)
Still a deterministic, zero-network rule/template composer -- no new hardcoded
question-to-answer pairs were added; every new composer (`compose_greeting_answer`,
`compose_comparison_answer`, `compose_missing_features_answer`,
`compose_structured_filter_answer`) is a pure function of structured evidence
(retrieved chunks, tool results, extracted slots), the same shape the original
`compose_policy_answer`/`compose_return_risk_answer` already were.

### Optional live LLM
`part3/live_llm.py` was not modified -- it already matched the target
retrieval-then-rewrite architecture (evidence -> MOCK_LLM answer -> optional
phrasing-only rewrite via local Ollama, `source`/`confidence` always the deterministic
values). A local Ollama install was found on this machine, but only a 26B model
(`gemma4:26b`), not the documented default (`llama3.2:3b`); a live smoke test was
skipped to avoid a slow, heavy invocation of an unrelated model -- this path is
optional and never scored either way.

## 3. Unseen-question evaluation

`scripts/run_chatbot_upgrade_eval.py` runs 25 natural-language queries -- none present
in the few-shot exemplars, `DEMO_TURNS`, or any committed transcript -- through the real
graph and classifies each row mechanically from the actual returned state. Full results:
[`reports/chatbot_upgrade_evaluation.md`](chatbot_upgrade_evaluation.md).

| category | count |
|---|---:|
| safe_refusal | 7 |
| grounded_response | 6 |
| correctly_understood (conversational) | 4 |
| correct_retrieval (multi-doc / structured filter) | 4 |
| clarification_requested | 2 |
| correct_tool_selected | 2 |

Zero rows classified `incorrect`. The seven `safe_refusal` rows are a mix of genuinely
out-of-domain questions (Flipkart's GST number), a blocked injection attempt, and
honestly-declined ungrounded policy questions -- none is a wrong tool selection or a
fabricated answer.

## 4. Tests

- `pytest`: **55 passed** (7 Part 1 + 6 Part 2 + 23 Part 3, all pre-existing and
  unmodified, + 19 new in `tests/test_agent_upgrade.py`).
- `tests/test_agent_upgrade.py` uses only phrasings absent from every exemplar/demo/
  transcript list: greeting/help routing without misfire, comparison grounding on ≥2
  distinct documents, multi-part questions, typo tolerance, slot-filling accumulation
  and scoped clarification, a completed slot-filled risk score, risk-explanation
  wording, structured catalog filtering against the real catalog count, catalog fuzzy
  search, catalog non-contamination of an ordinary own-order question, injection
  blocking, determinism of every new composer, product-catalog generation determinism,
  and zero-network-calls for every new path.
- `python3 validate_project.py`: **37/37** -- all Part 1/2 checks untouched; all Part 3
  checks (KB size, FAISS index, ≥8 MOCK_LLM transcripts, retrieval eval report) still
  hold.

## 5. Guardrails

Unchanged and re-verified: `part3/guardrails.py`'s structural regex patterns still block
genuine injection attempts (`test_injection_attempt_never_seen_in_existing_tests_is_
still_blocked`, a phrase never in the original test suite) and still don't false-positive
on ordinary questions sharing trigger words
(`test_ordinary_question_sharing_injection_vocabulary_is_not_blocked`). The new
conversational/comparison/catalog paths were confirmed to make zero network calls with
sockets poisoned, exactly like the original MOCK_LLM guarantee.

## 6. Assignment compliance

Nothing required was removed or altered in a way that breaks grading: the 6,000-row
`orders_dataset.csv` and its generator are untouched; Part 1's baseline/LogReg/
threshold-sweep/RandomForest+GridSearchCV/`t*_rf`/feature-importance/subgroup-analysis
pipeline is untouched; Part 2's Fashion-MNIST/transfer-learning/confusion-matrix
pipeline is untouched; the required `data/policy_index/` FAISS index, 16-document KB,
sentence-level chunking with parent-document IDs, `all-MiniLM-L6-v2` embeddings,
Precision@3/Recall@3 evaluation, LangGraph's five nodes with one real conditional edge,
MOCK_LLM as the default zero-network mode, guardrails, 8 transcripts, and the full test
suite are all untouched or purely extended.

## 7. Commands

```bash
# Launch (optional API + frontend console)
python3 scripts/generate_product_catalog.py   # regenerate the 55-SKU catalog (deterministic)
python3 -m part3.build_product_index          # build the catalog's own FAISS index
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8010
cd frontend && npm install && npm run dev     # http://localhost:5173

# Tests
pytest                                        # 55 tests across Parts 1-3 + the upgrade

# Validation
python3 validate_project.py                   # 37 acceptance checks
python3 scripts/run_chatbot_upgrade_eval.py   # regenerates chatbot_upgrade_evaluation.md
```

## 8. Demonstrating this in a viva

1. Open the Assistant screen and type something never shown as a suggestion --
   e.g. "my package arrived late and I paid cash on delivery, what should I know?" --
   to show it isn't a fixed question menu.
2. Say "hi" to show the conversational lane (previously a refusal).
3. Ask "Compare footwear and electronics return policies." to show multi-document
   grounding (expand "N retrieved document(s)" to show two distinct source IDs).
4. Run the slot-filling sequence: "My order is for running shoes and cost 4500 rupees,
   paid by COD." -> "What is the return risk?" -> "It took 6 days to arrive." -- shows
   accumulation across turns, a scoped clarification, and a completed risk score with an
   explanation grounded in Part 1's real permutation-importance report.
5. Ask "Which products are non-returnable?" to show the structured catalog filter
   returning an exact, real count -- then open `reports/chatbot_upgrade_evaluation.md`
   to show the 25-query unseen-question audit trail.
