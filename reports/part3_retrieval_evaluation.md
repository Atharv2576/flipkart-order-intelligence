# Part 3 -- Retrieval evaluation (Task 10)

Document-level Precision@3 / Recall@3.

## "How many days do I have to return a mobile phone?"

- relevant: ['POL03_electronics_return_window']
- retrieved (top-3 chunks, deduped to documents): ['POL03_electronics_return_window', 'POL04_home_return_window', 'POL05_beauty_return_window']
- hits: ['POL03_electronics_return_window']
- precision = 1/3 = **0.3333**
- recall = 1/1 = **1.0**

## "When will I get my refund if I paid cash on delivery?"

- relevant: ['POL06_cod_refund_timeline']
- retrieved (top-3 chunks, deduped to documents): ['POL06_cod_refund_timeline']
- hits: ['POL06_cod_refund_timeline']
- precision = 1/1 = **1.0**
- recall = 1/1 = **1.0**

## "Can I get a courier to pick up my return, or do I have to ship it myself?"

- relevant: ['POL10_reverse_pickup_eligibility']
- retrieved (top-3 chunks, deduped to documents): ['POL01_apparel_return_window', 'POL10_reverse_pickup_eligibility', 'POL11_reverse_pickup_process']
- hits: ['POL10_reverse_pickup_eligibility']
- precision = 1/3 = **0.3333**
- recall = 1/1 = **1.0**

## "My package arrived to a non-metro address, how long should delivery take?"

- relevant: ['POL08_delivery_sla']
- retrieved (top-3 chunks, deduped to documents): ['POL08_delivery_sla', 'POL09_delayed_delivery']
- hits: ['POL08_delivery_sla']
- precision = 1/2 = **0.5**
- recall = 1/1 = **1.0**

## "The shoes I received are damaged, what should I do?"

- relevant: ['POL12_damaged_product']
- retrieved (top-3 chunks, deduped to documents): ['POL02_footwear_return_window']
- hits: []
- precision = 0/1 = **0.0**
- recall = 0/1 = **0.0**

## "Can I exchange a dress for a bigger size instead of returning it?"

- relevant: ['POL15_exchange_policy']
- retrieved (top-3 chunks, deduped to documents): ['POL01_apparel_return_window', 'POL15_exchange_policy']
- hits: ['POL15_exchange_policy']
- precision = 1/2 = **0.5**
- recall = 1/1 = **1.0**

## "Is it too late to cancel my order if it hasn't shipped yet?"

- relevant: ['POL14_order_cancellation']
- retrieved (top-3 chunks, deduped to documents): ['POL09_delayed_delivery', 'POL14_order_cancellation']
- hits: ['POL14_order_cancellation']
- precision = 1/2 = **0.5**
- recall = 1/1 = **1.0**

**Average Precision@3: 0.4524**

**Average Recall@3: 0.8571**