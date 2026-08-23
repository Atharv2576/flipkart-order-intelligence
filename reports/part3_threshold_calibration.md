# Part 3 -- Groundedness threshold calibration

## In-domain queries (should be answerable)

- 0.6357 -- "How many days do I have to return a mobile phone?"
- 0.6807 -- "Can I return a beauty product I already opened?"
- 0.7174 -- "How long does a prepaid refund take to reach my card?"
- 0.5687 -- "What happens if my delivery is running late?"
- 0.8051 -- "Can I exchange footwear for a different size?"

## Out-of-domain queries (should be refused)

- 0.4462 -- "What is Flipkart's GST registration number?"
- 0.1468 -- "Who won the cricket match last night?"
- 0.1447 -- "What's the weather like in Bangalore today?"
- 0.0735 -- "Can you recommend a good recipe for biryani?"
- 0.1372 -- "What is the capital of France?"

- lowest in-domain score: **0.5687**
- highest out-of-domain score: **0.4462**
- separation gap: **0.1225**
- **threshold chosen: 0.5** (inside the gap)

Every in-domain query above clears this threshold; every out-of-domain query falls below it.