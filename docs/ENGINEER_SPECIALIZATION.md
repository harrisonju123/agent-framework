# Engineer Specialization System

The Agent Framework automatically selects specialized engineer profiles based on task file patterns, providing domain-specific guidance, teammates, and tool recommendations.

## Overview

When an engineer agent picks up a task, the framework analyzes the files involved and selects one of three specialized profiles:

- **Backend Engineer** - Server-side APIs, databases, business logic
- **Frontend Engineer** - UI components, state management, accessibility
- **Infrastructure Engineer** - DevOps, containers, CI/CD, IaC

If no clear specialization is detected, the engineer operates in generic mode with the base prompt.

## How It Works

### 1. File Pattern Detection

The system extracts file paths from multiple sources:

- `task.plan.files_to_modify` - Primary source from architect's plan
- `task.context.files` - Explicit file list in context
- `task.context.structured_findings` - Files from code review findings
- Task description text - Parses file paths like `src/api/handler.go`

### 2. Profile Matching

Each profile defines glob patterns that match relevant file types:

**Backend patterns:**
```
**/*.go, **/*.py, **/*.rb, **/*.java
**/api/**, **/models/**, **/services/**
**/*_test.go, **/*_test.py
```

**Frontend patterns:**
```
**/*.tsx, **/*.jsx, **/*.vue, **/*.svelte
**/*.css, **/*.scss
**/components/**, **/pages/**
```

**Infrastructure patterns:**
```
Dockerfile*, docker-compose*.yml
**/*.tf, **/*.tfvars
.github/workflows/**, k8s/**
```

### 3. Scoring and Selection

The system counts how many task files match each profile's patterns. A profile is selected if:

- It has the highest match count
- It matches ≥2 files
- It matches >50% of total files

This ensures a clear signal before applying specialization.

### 4. Prompt Augmentation

The selected profile's `prompt_suffix` is appended to the base engineer prompt, adding:

- Domain-specific focus areas
- Best practices for that domain
- Common pitfalls to avoid
- Language/framework-specific guidance

### 5. Specialized Teammates

The profile's teammates are merged with the engineer's base teammates:

**Backend teammates:**
- `database-expert` - Reviews queries, indexes, migrations
- `api-reviewer` - Checks REST conventions, error handling

**Frontend teammates:**
- `ux-reviewer` - Accessibility, responsive design, user experience
- `performance-auditor` - Bundle size, re-renders, optimization

**Infrastructure teammates:**
- `security-hardening` - Secret management, permissions, vulnerabilities
- `sre-reviewer` - Monitoring, logging, reliability

## Configuration

### Using the Default Profiles

By default, the system uses hardcoded profiles with sensible defaults. No configuration required.

### Customizing via YAML

Create `config/specializations.yaml` to override or extend profiles:

```yaml
enabled: true  # Global toggle

profiles:
  - id: backend
    name: Backend Engineer
    description: Server-side specialist
    file_patterns:
      - "**/*.go"
      - "**/*.py"
      - "**/api/**"
    prompt_suffix: |
      BACKEND SPECIALIZATION:
      Focus on API design, database optimization, and error handling.

      Best practices:
      - Use parameterized queries
      - Implement proper logging
      - Write integration tests

    tool_guidance: |
      Common tools: go test, pytest, postman, psql

    teammates:
      database-expert:
        description: "Reviews database code"
        prompt: "Check for N+1 queries, missing indexes, migration safety."
```

### Global Enable/Disable

Disable specialization entirely via `config/specializations.yaml`:

```yaml
enabled: false
profiles: []
```

### Per-Agent Toggle

Disable specialization for specific engineer agents in `config/agents.yaml`:

```yaml
agents:
  - id: engineer
    name: Software Engineer
    specialization_enabled: false  # Skip specialization for this agent
    # ... rest of config
```

## Monorepo Support: Specialization Hints

For monorepos where file patterns are ambiguous, tasks can explicitly specify a profile:

```python
task = Task(
    id="fix-backend-api",
    type=TaskType.FIX,
    assigned_to="engineer",
    context={
        "specialization_hint": "backend"  # Force backend profile
    },
    # ...
)
```

Valid hints: `"backend"`, `"frontend"`, `"infrastructure"`

Invalid hints are logged and fall back to file-based detection.

## Examples

### Backend Task

**Input:**
```python
task.plan.files_to_modify = [
    "cmd/server/main.go",
    "internal/api/handler.go",
    "internal/service/user.go",
]
```

**Result:**
- Profile: Backend Engineer
- Prompt includes: API design, database best practices, error handling
- Teammates: `database-expert`, `api-reviewer`

### Frontend Task

**Input:**
```python
task.plan.files_to_modify = [
    "src/components/UserProfile.tsx",
    "src/pages/Dashboard.tsx",
    "src/styles/dashboard.scss",
]
```

**Result:**
- Profile: Frontend Engineer
- Prompt includes: Component design, accessibility, performance optimization
- Teammates: `ux-reviewer`, `performance-auditor`

### Infrastructure Task

**Input:**
```python
task.plan.files_to_modify = [
    "Dockerfile",
    "k8s/deployment.yaml",
    ".github/workflows/deploy.yml",
]
```

**Result:**
- Profile: Infrastructure Engineer
- Prompt includes: Container best practices, IaC, security hardening
- Teammates: `security-hardening`, `sre-reviewer`

### Mixed Task (No Specialization)

**Input:**
```python
task.plan.files_to_modify = [
    "README.md",
    "docs/guide.txt",
]
```

**Result:**
- Profile: None (generic engineer)
- Uses base engineer prompt unchanged
- Base teammates only

## Observability

### Logging

Specialization detection logs include:

```
INFO: Selected specialization 'Backend Engineer' (score=3, threshold=2.0, total_files=3)
```

```
DEBUG: Profile 'backend': 3/3 files matched (100%)
DEBUG: Profile 'frontend': 0/3 files matched (0%)
```

### Activity Tracking

The `AgentActivity.specialization` field tracks the selected profile for each task, recorded when prompt building detects a specialization match.

## Troubleshooting

### Specialization Not Detected

**Symptom:** Engineer uses generic prompt despite having specialized files.

**Causes:**
1. Too few matching files (need ≥2)
2. Mixed file types with no clear winner (e.g., 1 backend + 1 frontend)
3. Specialization disabled globally or per-agent

**Solution:**
- Check logs for scoring: `Profile 'backend': X/Y files matched`
- Verify `specialization_enabled: true` in `config/agents.yaml`
- Use `specialization_hint` for ambiguous cases

### Wrong Profile Selected

**Symptom:** Backend task gets frontend profile.

**Causes:**
1. File patterns are ambiguous (e.g., `.ts` files match both backend Node.js and frontend)
2. Task includes files from multiple domains

**Solution:**
- Use `specialization_hint` to force correct profile
- Review file patterns in your `config/specializations.yaml`
- Check logs: `Profile 'frontend': 2/2 files matched (100%)`

### Custom Profile Not Loading

**Symptom:** YAML profile changes not reflected.

**Causes:**
1. Syntax error in YAML
2. Config cache not cleared (dev environment)
3. File path incorrect

**Solution:**
- Validate YAML syntax
- Check logs: `Loaded N specialization profiles from YAML`
- Ensure file is at `config/specializations.yaml` (relative to workspace root)

## Testing

The specialization system includes comprehensive test coverage:

```bash
# Unit tests
pytest tests/unit/test_engineer_specialization.py -v

# Integration tests
pytest tests/integration/test_specialization_integration.py -v
```

Test coverage:
- File pattern detection from all sources
- Profile matching and scoring
- Prompt augmentation
- Teammate merging
- YAML config loading
- Enable/disable toggles
- Specialization hints
- Edge cases (mixed files, no files, etc.)

## Best Practices

1. **Let File Patterns Guide Selection**
   Trust the automatic detection for most tasks. Only use hints when patterns are truly ambiguous.

2. **Keep Prompt Extensions Focused**
   Specialization prompts should be <500 chars of high-signal guidance. Avoid generic advice.

3. **Use Specialized Teammates Judiciously**
   Each teammate adds overhead. Only include teammates that provide unique value for the domain.

4. **Test Custom Profiles**
   After modifying `specializations.yaml`, run tasks through each profile to verify prompt quality.

5. **Document Profile Intent**
   Use clear `description` fields in YAML so future maintainers understand profile purpose.

## Limitations

1. **Single Profile Per Task**
   Tasks can't mix profiles (e.g., "backend + infrastructure"). Workaround: use hints or split into subtasks.

2. **Static Pattern Matching**
   Detection is based on file extensions/paths only, not file content. A Python file in `frontend/` is still backend.

3. **No Learning**
   Profiles don't adapt based on task outcomes. Patterns must be manually tuned.

4. **English-Only Prompts**
   Specialization prompts are currently English-only.

## Future Enhancements

- **Multi-Profile Support**: Allow tasks to use multiple profiles simultaneously
- **Dynamic Profile Learning**: Adjust patterns based on task success rates
- **Custom Profiles Per Repository**: Repository-specific specializations via `.agent-profiles.yaml`
- **Specialization Metrics**: Track profile effectiveness and selection accuracy
- **Content-Based Detection**: Analyze file content, not just paths/extensions
- **Localized Prompts**: Multi-language support for specialization guidance

## Reference

### Key Files

- `src/agent_framework/core/engineer_specialization.py` - Core logic
- `src/agent_framework/core/config.py` - Config models (`SpecializationConfig`, `SpecializationProfileConfig`)
- `config/specializations.yaml` - Profile definitions (optional)
- `config/agents.yaml` - Per-agent toggle (`specialization_enabled`)

### Key Functions

- `detect_specialization(task)` - Select profile for a task
- `apply_specialization_to_prompt(prompt, profile)` - Augment prompt with profile context
- `get_specialized_teammates(base, profile)` - Merge teammates
- `get_specialization_enabled()` - Check global enable flag

### Data Models

```python
@dataclass
class SpecializationProfile:
    id: str                           # "backend", "frontend", "infrastructure"
    name: str                         # "Backend Engineer"
    description: str                  # Profile purpose
    file_patterns: List[str]          # Glob patterns to match
    prompt_suffix: str                # Additional guidance
    teammates: Dict[str, Dict]        # Specialized teammates
    tool_guidance: str                # Tool recommendations
```
