# Engineer Specialization Implementation Plan

## Overview
Enable engineer agents to specialize based on task file patterns, providing domain-specific prompts, teammates, and tool guidance.

## Problem Statement
Currently, all engineer agents receive the same generic prompt regardless of whether they're working on:
- Backend code (APIs, databases, services)
- Frontend code (UI components, state management)
- Infrastructure (Docker, Kubernetes, CI/CD)

This results in:
- Generic guidance that misses domain-specific best practices
- Inability to configure specialized teammates (e.g., db-reviewer for backend)
- Suboptimal tool recommendations

## Solution Design

### 1. Profile System Architecture

Create a specialization system that:
1. Defines engineer profiles with file patterns, prompt extensions, and teammates
2. Automatically selects the best profile based on task file patterns
3. Augments the base engineer prompt with specialization guidance
4. Composes specialized teammate teams

### 2. Data Models

```python
# src/agent_framework/core/config.py

class EngineerProfile(BaseModel):
    """Engineer specialization profile."""
    id: str  # "backend", "frontend", "infra"
    name: str  # "Backend Engineer", "Frontend Engineer", etc.
    file_patterns: List[str]  # ["**/*.py", "**/*.go", "**/api/**"]
    prompt_extension: str  # Domain-specific guidance
    teammates: Dict[str, TeammateDefinition]  # Specialized teammates
    tool_guidance: Optional[str] = None  # Tool recommendations
```

### 3. Configuration Format

```yaml
# config/agents.yaml

agents:
  - id: engineer
    name: Software Engineer
    queue: engineer
    enabled: true
    prompt: |
      [Base engineer prompt - unchanged]

    # NEW: Specialization profiles
    engineer_profiles:
      backend:
        name: Backend Engineer
        file_patterns:
          - "**/*.py"
          - "**/*.go"
          - "**/*.java"
          - "**/*.rb"
          - "**/migrations/**"
          - "**/models/**"
          - "**/api/**"
        prompt_extension: |
          SPECIALIZATION: Backend Engineering

          Focus areas:
          - API design and REST conventions
          - Database queries and migrations
          - Performance and concurrency
          - Error handling and retries
          - Background jobs and queues

          When implementing:
          - Use parameterized queries to prevent SQL injection
          - Add database indexes for query performance
          - Handle errors gracefully with proper status codes
          - Consider rate limiting and caching strategies

        teammates:
          db-reviewer:
            description: "Database specialist - reviews queries, migrations, indexing"
            prompt: "You review database code. Check for N+1 queries, missing indexes, migration safety, and query optimization opportunities."
          api-designer:
            description: "API design specialist - reviews REST conventions, OpenAPI specs"
            prompt: "You review API design. Check for REST best practices, consistent error handling, proper status codes, and clear documentation."

        tool_guidance: "Prefer pytest, go test, postman, database tools"

      frontend:
        name: Frontend Engineer
        file_patterns:
          - "**/*.tsx"
          - "**/*.jsx"
          - "**/*.vue"
          - "**/*.svelte"
          - "**/components/**"
          - "**/pages/**"
          - "**/styles/**"
        prompt_extension: |
          SPECIALIZATION: Frontend Engineering

          Focus areas:
          - UI/UX and accessibility (WCAG 2.1)
          - Browser compatibility
          - State management patterns
          - Performance (bundle size, rendering)
          - Responsive design

          When implementing:
          - Use semantic HTML for accessibility
          - Add ARIA labels where needed
          - Test keyboard navigation
          - Optimize bundle size (code splitting, lazy loading)
          - Handle loading and error states

        teammates:
          ux-reviewer:
            description: "UX specialist - reviews accessibility, responsive design"
            prompt: "You review UI/UX. Check for accessibility issues, responsive design, consistent styling, and good user experience patterns."
          perf-auditor:
            description: "Performance specialist - reviews bundle size, rendering"
            prompt: "You review frontend performance. Check for large bundles, unnecessary re-renders, missing memoization, and inefficient queries."

        tool_guidance: "Prefer jest, cypress, playwright, webpack analyzer, lighthouse"

      infra:
        name: Infrastructure Engineer
        file_patterns:
          - "**/Dockerfile"
          - "**/*.tf"
          - "**/*.yaml"
          - "**/*.yml"
          - "**/k8s/**"
          - "**/terraform/**"
          - "**/.github/**"
          - "**/helm/**"
        prompt_extension: |
          SPECIALIZATION: Infrastructure Engineering

          Focus areas:
          - Container security and optimization
          - Infrastructure as Code (Terraform, Helm)
          - CI/CD pipeline design
          - Observability (metrics, logs, traces)
          - Scalability and reliability

          When implementing:
          - Use minimal base images (alpine, distroless)
          - Never commit secrets - use secret managers
          - Add health checks to all services
          - Implement proper logging and metrics
          - Consider failure scenarios and recovery

        teammates:
          security-scanner:
            description: "Security specialist - reviews container security, secrets"
            prompt: "You review infrastructure security. Check for exposed secrets, vulnerable base images, overly permissive IAM policies, and security best practices."
          reliability-engineer:
            description: "Reliability specialist - reviews monitoring, alerting"
            prompt: "You review system reliability. Check for missing health checks, inadequate monitoring, poor error handling, and failure recovery mechanisms."

        tool_guidance: "Prefer docker, terraform, kubectl, helm, trivy (security scanning)"
```

### 4. Profile Selection Logic

```python
# src/agent_framework/core/config.py

def select_engineer_profile(
    task_context: dict,
    agent_def: AgentDefinition
) -> Optional[EngineerProfile]:
    """Select best engineer profile based on task file patterns.

    Returns:
        EngineerProfile if match found, None for generic engineer
    """
    if not hasattr(agent_def, 'engineer_profiles') or not agent_def.engineer_profiles:
        return None

    # Extract file patterns from task context
    files = task_context.get("files_to_modify", [])
    if not files:
        # Check alternate locations
        files = task_context.get("affected_files", [])
    if not files:
        return None

    # Score each profile by pattern matches
    scores = {}
    for profile_id, profile in agent_def.engineer_profiles.items():
        matches = 0
        for file in files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in profile.file_patterns):
                matches += 1
        if matches > 0:
            scores[profile_id] = matches

    # Return profile with highest score
    if scores:
        best_profile_id = max(scores, key=scores.get)
        return agent_def.engineer_profiles[best_profile_id]

    return None
```

### 5. Integration Points

#### A. run_agent.py
Detect specialization and augment prompt:

```python
# After loading agent_def, before creating Agent
selected_profile = None
if base_agent_id == "engineer":
    # Load task to check for file patterns
    task_file = workspace / framework_config.communication_dir / "queues" / agent_def.queue / "current_task.json"
    if task_file.exists():
        with open(task_file) as f:
            task_data = json.load(f)
            selected_profile = select_engineer_profile(task_data.get("context", {}), agent_def)

# Augment prompt if profile selected
agent_prompt = agent_def.prompt
if selected_profile:
    agent_prompt = f"{agent_def.prompt}\n\n{selected_profile.prompt_extension}"
    logger.info(f"Engineer specialization: {selected_profile.name}")

agent_config = AgentConfig(
    id=agent_id,
    name=agent_name,
    queue=agent_def.queue,
    prompt=agent_prompt,  # Use augmented prompt
    # ... rest unchanged
)
```

#### B. team_composer.py
Use specialized teammates:

```python
def compose_default_team(
    agent_def: AgentDefinition,
    default_model: str = "sonnet",
    selected_profile: Optional[EngineerProfile] = None,
) -> Optional[dict]:
    """Build team from agent's configured teammates or profile teammates."""

    # If engineer profile selected, use its teammates
    if selected_profile and hasattr(selected_profile, 'teammates'):
        agents_dict = {}
        for teammate_id, teammate in selected_profile.teammates.items():
            agents_dict[teammate_id] = {
                "model": teammate.model or default_model,
                "description": teammate.description,
                "prompt": teammate.prompt,
            }
        return agents_dict if agents_dict else None

    # Otherwise use agent's default teammates
    if not agent_def.teammates:
        return None

    # ... existing logic
```

#### C. activity.py
Track specialization for observability:

```python
class AgentActivity(BaseModel):
    # ... existing fields
    specialization: Optional[str] = None  # "backend", "frontend", "infra", None
```

### 6. Implementation Steps

1. **Add EngineerProfile model to config.py** (~50 lines)
   - Define Pydantic model with validation
   - Add to AgentDefinition as optional field

2. **Add profile selection logic to config.py** (~40 lines)
   - Implement select_engineer_profile function
   - Add fnmatch-based pattern matching
   - Handle edge cases (no files, multiple matches)

3. **Update agents.yaml with profiles** (~120 lines)
   - Add engineer_profiles section
   - Define backend, frontend, infra profiles
   - Configure teammates for each

4. **Modify run_agent.py for specialization** (~30 lines)
   - Detect task file patterns
   - Select profile and augment prompt
   - Log selected specialization

5. **Update team_composer.py** (~40 lines)
   - Accept selected_profile parameter
   - Use profile teammates when available
   - Maintain backward compatibility

6. **Add specialization to activity tracking** (~10 lines)
   - Add field to AgentActivity
   - Populate in agent initialization

7. **Create documentation** (~160 lines)
   - Configuration guide
   - Pattern matching examples
   - Troubleshooting guide

## Backward Compatibility

- Tasks without file patterns → generic engineer (current behavior)
- Agents without engineer_profiles → no specialization
- All existing workflows continue unchanged

## Success Metrics

1. Backend tasks with `*.py, *.go` files → backend engineer + db-reviewer teammate
2. Frontend tasks with `*.tsx, *.jsx` files → frontend engineer + ux-reviewer teammate
3. Infra tasks with `Dockerfile, *.tf` files → infra engineer + security-scanner teammate
4. Tasks without patterns → generic engineer (unchanged)
5. Activity logs show selected specialization
6. All tests pass

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Ambiguous profile selection (multi-domain tasks) | Select profile with most file matches |
| Prompt length increase | Keep extensions <500 chars, focused guidance |
| Teammate configuration complexity | Teammates remain optional, graceful fallback |
| Breaking existing workflows | Default to generic engineer if no match |

## Testing Strategy

1. Unit tests for select_engineer_profile with various file patterns
2. Integration tests for each specialization path
3. Backward compatibility tests (no patterns = generic engineer)
4. Activity log verification tests

## Estimated Effort

- Total: ~450 lines of code + 160 lines documentation
- Complexity: Medium (new feature, backward compatible)
- Time: 1 implementation sprint

## Files to Modify

1. `src/agent_framework/core/config.py` - Models and selection logic (90 lines)
2. `config/agents.yaml` - Profile definitions (120 lines)
3. `src/agent_framework/run_agent.py` - Specialization detection (30 lines)
4. `src/agent_framework/core/team_composer.py` - Profile teammates (40 lines)
5. `src/agent_framework/core/activity.py` - Specialization tracking (10 lines)
6. `docs/ENGINEER_SPECIALIZATION.md` - Documentation (160 lines)

## Future Enhancements

1. Custom profiles per repository (repo-specific specializations)
2. Dynamic profile learning based on task outcomes
3. Multi-profile support (e.g., backend + infra hybrid)
4. Profile metrics and effectiveness tracking
