"""The retrieval-evaluation answer key (Task 10) and the in-domain /
out-of-domain query sets used to calibrate the groundedness threshold.
Every pair here is document-level: which document(s) a human would judge
relevant to the query, not which chunk.
"""

# query -> set of relevant document_ids (a query may have more than one
# genuinely relevant document, e.g. a question that a related policy also
# touches on).
RETRIEVAL_EVAL_QUERIES = [
    {
        "query": "How many days do I have to return a mobile phone?",
        "relevant_document_ids": {"POL03_electronics_return_window"},
    },
    {
        "query": "When will I get my refund if I paid cash on delivery?",
        "relevant_document_ids": {"POL06_cod_refund_timeline"},
    },
    {
        "query": "Can I get a courier to pick up my return, or do I have to ship it myself?",
        "relevant_document_ids": {"POL10_reverse_pickup_eligibility"},
    },
    {
        "query": "My package arrived to a non-metro address, how long should delivery take?",
        "relevant_document_ids": {"POL08_delivery_sla"},
    },
    {
        "query": "The shoes I received are damaged, what should I do?",
        "relevant_document_ids": {"POL12_damaged_product"},
    },
    {
        "query": "Can I exchange a dress for a bigger size instead of returning it?",
        "relevant_document_ids": {"POL15_exchange_policy"},
    },
    {
        "query": "Is it too late to cancel my order if it hasn't shipped yet?",
        "relevant_document_ids": {"POL14_order_cancellation"},
    },
]

# Genuine policy questions the knowledge base can answer -- used to find the
# lowest score a real in-domain question should still clear.
IN_DOMAIN_CALIBRATION_QUERIES = [
    "How many days do I have to return a mobile phone?",
    "Can I return a beauty product I already opened?",
    "How long does a prepaid refund take to reach my card?",
    "What happens if my delivery is running late?",
    "Can I exchange footwear for a different size?",
]

# Questions the knowledge base has no business answering -- used to find the
# highest score an out-of-domain question reaches, so the threshold can sit
# in the gap between the two.
OUT_OF_DOMAIN_CALIBRATION_QUERIES = [
    "What is Flipkart's GST registration number?",
    "Who won the cricket match last night?",
    "What's the weather like in Bangalore today?",
    "Can you recommend a good recipe for biryani?",
    "What is the capital of France?",
]
