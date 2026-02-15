# Multi-Perspective Debate

## When to Use
- Architectural decisions with significant trade-offs
- Technology/library choices
- Design pattern selection
- Risk assessment for major changes
- Any decision where both options have legitimate merits

## When NOT to Use
- Routine implementation decisions
- Questions with a clear best answer
- Quick factual questions (use consult_agent instead)

## How It Works
1. You provide a topic and optional context
2. An Advocate argues FOR the proposed approach
3. A Critic argues AGAINST and proposes alternatives
4. An Arbiter synthesizes both arguments into a recommendation

## Cost
Each debate uses 2 consultation slots (out of 5 per session).

## Example
```
debate_topic(
  topic="Should we add Redis caching for the user API?",
  context="Current response times are 200ms, target is <100ms. 500 requests/sec. PostgreSQL backend."
)
```

Returns structured result with:
- advocate_argument: Why caching helps
- critic_argument: Risks, alternatives (query optimization, connection pooling)
- synthesis: Arbiter's recommendation
- confidence: high/medium/low
- trade_offs: Key points both sides agreed matter
