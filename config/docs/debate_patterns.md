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

```typescript
debate_topic(
  topic="Should we add Redis caching for the user API?",
  context="Current response times are 200ms, target is <100ms. 500 requests/sec. PostgreSQL backend."
)
```

Returns structured result with:
- `advocate_argument`: Why caching helps
- `critic_argument`: Risks, alternatives (query optimization, connection pooling)
- `synthesis`: Arbiter's recommendation
- `confidence`: high/medium/low
- `trade_offs`: Key points both sides agreed matter

## Decision Framework

Use debate_topic when:
- **Multiple valid solutions exist** with different strengths/weaknesses
- **Irreversible or costly to change** after implementation
- **Team alignment matters** and you want structured analysis

Skip debate_topic when:
- **One solution is clearly superior** (ask consult_agent instead)
- **Low stakes** decision that can be easily changed later
- **Consultation budget is tight** (debate costs 2 slots vs consult's 1)

## Best Practices

### Framing the Topic

**Good**: Specific, actionable questions with clear alternatives
- "Should we use gRPC or REST for the inter-service communication API?"
- "Is adding an index on user_id worth the write overhead?"

**Bad**: Vague or overly broad questions
- "How do we improve performance?" (no specific approach to debate)
- "What's the best database?" (too open-ended, no context)

### Providing Context

Include just enough context for informed debate:
- Current state and constraints
- Performance/scale requirements
- Team expertise or existing patterns

Avoid:
- Overly detailed implementation specs (>1000 chars get truncated)
- Bias-inducing phrasing (let debaters form their own views)

### Custom Positions

Use `advocate_position` and `critic_position` to override defaults:

```typescript
debate_topic(
  topic="Migrate from MySQL to PostgreSQL?",
  advocate_position="in favor of PostgreSQL for JSON support and performance",
  critic_position="against migration due to operational risk and team learning curve"
)
```

Useful when you want to test a specific angle or constraint.

## Output Interpretation

### Confidence Levels

- **High**: Clear winner, both perspectives agree on direction
- **Medium**: Trade-offs are balanced, context-dependent choice
- **Low**: Insufficient information, or arbiter couldn't synthesize

If confidence is low, consider:
- Adding more context and re-running debate
- Consulting a subject matter expert (consult_agent with architect/engineer)
- Prototyping both approaches before committing

### Trade-offs

The `trade_offs` array highlights key considerations that apply regardless of which option you choose:

```json
{
  "trade_offs": [
    "Adding caching increases system complexity",
    "Cache invalidation must be carefully managed",
    "Redis adds operational overhead but significantly improves read performance"
  ]
}
```

Use these to:
- Plan for risks in your implementation
- Document decision rationale in ADRs or JIRA tickets
- Set expectations with stakeholders

### Recommendation

The arbiter's `recommendation` is a one-sentence verdict:
- "Proceed with Redis caching for the user API"
- "Optimize PostgreSQL queries before adding caching layer"
- "Prototype both approaches to measure real-world impact"

Treat it as a strong suggestion, not gospel. Your domain knowledge and project constraints should inform the final call.

## Example Workflows

### Architect Planning Phase

```typescript
// Before breaking down a large feature
const result = await debate_topic(
  topic="Should we implement real-time notifications with WebSockets or SSE?",
  context="Browser-only clients, need bidirectional for future chat feature, 10k concurrent users expected"
);

// Use result.recommendation to inform plan.approach
// Include result.trade_offs in plan.risks
```

### Engineer Implementation Dilemma

```typescript
// Stuck on performance vs readability trade-off
const result = await debate_topic(
  topic="Should we inline this hot-path function or keep it abstracted?",
  context="Function called 1M times/sec, current abstraction adds 10% overhead"
);

// If confidence is "high", proceed with recommendation
// If "medium" or "low", consult architect for guidance
```

### QA Severity Assessment

```typescript
// Unsure if a finding blocks the PR
const result = await debate_topic(
  topic="Is this SQL query performance issue CRITICAL or HIGH severity?",
  context="Query takes 500ms on 10k rows, worst-case 100k rows. User-facing API endpoint."
);

// Use result.confidence and result.recommendation to assign severity
```

## Troubleshooting

### "Debate limit reached"

You've exhausted your consultation budget (5 per session). Options:
- Proceed with your best judgment
- Consult shared knowledge base for prior decisions: `get_knowledge(topic="architectural-decisions")`
- Document the decision and move forward (can revisit in review)

### Both perspectives failed

Rare, but indicates Claude CLI issues or malformed prompts. Fall back to:
1. `consult_agent` with a simpler question
2. Documenting the decision rationale manually
3. Escalating to human review if it's a critical decision

### Arbiter returned low confidence

The debate didn't surface a clear winner. Next steps:
- Add more context (performance numbers, team constraints) and retry
- Use `consult_agent(target_agent="architect")` for a single expert opinion
- Build a quick prototype to validate assumptions

## Related Tools

- **consult_agent**: Single perspective, costs 1 slot, faster for straightforward questions
- **share_knowledge / get_knowledge**: Document debate outcomes for future reference
- **queue_task_for_agent**: If debate reveals need for prototyping, queue a spike task
