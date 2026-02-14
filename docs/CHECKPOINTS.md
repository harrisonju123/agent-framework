# Workflow Checkpoints

Checkpoints provide configurable pause points in workflows where human approval is required before the workflow continues. This addresses the need for control over high-stakes changes while maintaining autonomous execution for routine tasks.

## Overview

The agent framework supports two execution modes:
- **Fully autonomous**: Workflows execute end-to-end without human intervention
- **Checkpoint-gated**: Workflows pause at specific steps for human approval

Checkpoints solve the problem of "no middle ground" -- previously, you either had to manually intervene or trust the system completely. Now you can configure strategic pause points for review.

## When to Use Checkpoints

Add checkpoints at workflow steps where:
- **Production deployments** require verification
- **Database migrations** need approval
- **High-risk code changes** should be reviewed
- **Financial operations** require manual confirmation
- **External API integrations** need validation

## Configuration

### Workflow Definition

Checkpoints are configured in workflow YAML files by adding a `checkpoint` field to any workflow step:

```yaml
workflows:
  production-deploy:
    description: "Deploy to production with approval gate"
    steps:
      architect:
        agent: architect
        next:
          - target: engineer

      engineer:
        agent: engineer
        checkpoint:
          message: "Review implementation before deploying to production"
          reason: "Production deployment requires manual approval"
        next:
          - target: qa

      qa:
        agent: qa
        next: []
    start_step: architect
```

### Checkpoint Fields

- **`message`** (required): Human-readable description shown when checkpoint is reached
- **`reason`** (optional): Explanation for why approval is needed (for audit trail)

## Usage

### Viewing Checkpoints

List all tasks currently awaiting approval:

```bash
agent approve
```

### Approving Checkpoints

Approve a specific checkpoint to allow the workflow to continue:

```bash
# Basic approval
agent approve chain-abc123-engineer

# Approval with message
agent approve chain-abc123-engineer -m "Reviewed implementation, looks good"
```

### Monitoring

Tasks at checkpoints show status `AWAITING_APPROVAL` in the status dashboard:

```bash
agent status --watch
```

## Architecture

### Task States

New task status: `AWAITING_APPROVAL`

Task model fields:
```python
checkpoint_reached: Optional[str]      # ID of checkpoint reached
checkpoint_message: Optional[str]      # Message to display
approved_at: Optional[datetime]        # When approved
approved_by: Optional[str]             # Who approved (OS user)
```

### Execution Flow

1. **Agent completes task** - Workflow executor checks for checkpoint
2. **Checkpoint detected** - Task marked as `AWAITING_APPROVAL`
3. **Task saved to checkpoint queue** - `.agent-communication/queues/checkpoints/`
4. **Workflow paused** - No further routing until approval
5. **Human approves** - Task re-queued to assigned agent with approval metadata
6. **Workflow continues** - Executor sees approval and proceeds to next step

## Best Practices

1. **Strategic Placement**: Don't checkpoint every step -- focus on high-risk transitions
2. **Clear Messages**: Explain what to review and why approval is needed
3. **Document Criteria**: Include approval criteria in checkpoint message
4. **Audit Trail**: Use approval messages to document review outcomes
5. **Monitor Queues**: Regularly check `agent approve` to avoid workflow bottlenecks

## Troubleshooting

### Checkpoint not triggering

Check workflow configuration:
```bash
cat config/workflows.yaml
```

### Task stuck at checkpoint

```bash
# List checkpoints
agent approve

# Check task details
ls -la .agent-communication/queues/checkpoints/

# Approve or investigate
agent approve <task-id>
```

### Approval not resuming workflow

Ensure agents are running:
```bash
agent status
agent start
```
