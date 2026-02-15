# Multi-Perspective Debate System

## Overview

The `debate_topic` MCP tool enables agents to reason through complex decisions using structured adversarial perspectives. Instead of a single consultation, it spawns three perspectives in sequence:

1. **Advocate** — argues in favor of an approach, focusing on benefits and opportunities
2. **Critic** — argues against the approach, focusing on risks and downsides
3. **Arbiter** — synthesizes both arguments into a final recommendation with trade-offs

## When to Use Debates

Use `debate_topic` for complex decisions with significant trade-offs:

- **Architectural choices**: "Should we use Redis or in-memory caching for session storage?"
- **Approach decisions**: "Should we refactor this now or defer until after the feature ships?"
- **Technology selection**: "Should we adopt GraphQL or stick with REST for this new API?"
- **Risk evaluation**: "Is it worth migrating to the new library version now?"

**Don't use debates for:**
- Simple yes/no questions with clear answers
- Decisions already made or constrained by requirements
- Questions better suited to a single expert consultation
- Time-sensitive decisions where speed matters more than deliberation

## Cost and Rate Limiting

Debates consume **2 consultation slots** (out of the default 5 per session):
- Each consultation costs 1 slot
- Each debate costs 2 slots (advocate + critic + arbiter synthesis)

This prevents overuse while still allowing 1-2 debates per agent session when needed.

## Usage

### Basic Debate

```typescript
debate_topic({
  topic: "Should we implement server-side rendering for this dashboard?",
  context: "Dashboard has 10+ charts, used by 500+ users daily, currently client-side only"
})
```

### Custom Perspectives

For nuanced decisions, you can specify custom perspectives instead of the default advocate/critic:

```typescript
debate_topic({
  topic: "How should we structure our microservices deployment?",
  context: "Team of 8 engineers, 3 services currently, expecting to grow to 12 services",
  custom_perspectives: {
    advocate: "Argue for Kubernetes with full orchestration",
    critic: "Argue for simpler Docker Compose setup"
  }
})
```

## Output Structure

The debate returns a structured result:

```typescript
{
  success: true,
  topic: "...",
  advocate: {
    perspective: "Advocate - argue in favor...",
    argument: "Server-side rendering would improve initial load time...",
    success: true
  },
  critic: {
    perspective: "Critic - argue against...",
    argument: "SSR adds complexity and increases server costs...",
    success: true
  },
  synthesis: {
    recommendation: "Implement SSR for the dashboard landing page only...",
    confidence: "medium",
    trade_offs: [
      "Improved perceived performance vs increased server load",
      "Better SEO vs higher infrastructure costs",
      "Faster initial paint vs more complex deployment"
    ],
    reasoning: "Partial SSR balances the benefits of faster initial load..."
  },
  debate_id: "debate-1234567890-abc123",
  consultations_used: 2,
  consultations_remaining: 3
}
```

## Confidence Levels

The Arbiter assigns a confidence level to its recommendation:

- **high**: Clear winner, trade-offs are manageable, strong consensus possible
- **medium**: Both approaches have merit, decision depends on priorities
- **low**: Significant unknowns, close call, or requires additional information

Use the confidence level to decide how much weight to give the recommendation.

## Graceful Degradation

The debate system handles failures gracefully:

- **One perspective fails**: Arbiter synthesizes using the successful perspective plus fallback reasoning
- **Both perspectives fail**: Returns a low-confidence recommendation to proceed with best judgment
- **Arbiter fails**: Returns both arguments with a generic synthesis
- **Rate limit reached**: Returns immediately without spawning perspectives

Even in failure modes, you get actionable output.

## Best Practices

1. **Frame topics clearly**: State the decision as a question or choice
2. **Provide context**: Include constraints, requirements, team size, scale
3. **Use sparingly**: Reserve debates for genuinely complex decisions
4. **Consider the recommendation**: Don't blindly follow it—use it to inform your decision
5. **Review trade-offs**: The trade-offs list often reveals considerations you missed

## Integration with Consultations

Both `consult_agent` and `debate_topic` share the same rate limit pool:
- Starting consultations: 5
- After 1 consultation: 4 remaining
- After 1 debate: 2 remaining
- After 2 consultations and 1 debate: 0 remaining

Plan your consultations and debates accordingly. If you expect to need multiple expert opinions, consider whether a debate is worth the cost.

## Examples

### Architectural Decision

```typescript
debate_topic({
  topic: "Should we use WebSockets or Server-Sent Events for real-time notifications?",
  context: "Browser-based app, 1000+ concurrent users, notifications are one-way (server to client)"
})

// Advocate: Argues for WebSockets (full-duplex, battle-tested, libraries available)
// Critic: Argues for SSE (simpler, HTTP-based, automatic reconnection)
// Arbiter: Recommends SSE due to one-way requirement, lower complexity, and automatic reconnection
```

### Refactoring Decision

```typescript
debate_topic({
  topic: "Should we refactor the authentication module now or after the Q2 feature launch?",
  context: "Auth code is working but hard to extend, Q2 feature needs auth changes, 2 weeks until launch"
})

// Advocate: Refactor now (cleaner foundation, easier feature implementation)
// Critic: Defer refactoring (launch risk, feature is priority, can refactor after)
// Arbiter: Recommends minimal refactoring of only the extension points needed for Q2, full refactor after launch
```

### Technology Selection

```typescript
debate_topic({
  topic: "Should we adopt Zod for runtime validation or continue with custom validators?",
  context: "Currently have 50+ custom validators, TypeScript codebase, team is familiar with custom approach",
  custom_perspectives: {
    advocate: "Argue for migrating to Zod incrementally",
    critic: "Argue for keeping custom validators and improving them"
  }
})
```

## Logging and Debugging

All debates are logged to `.agent-communication/debates/` with full arguments and synthesis:

```bash
cat .agent-communication/debates/debate-1234567890-abc123.json
```

Use these logs to review past decisions and understand the reasoning.
