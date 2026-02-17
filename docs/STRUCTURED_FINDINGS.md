# Structured QA Findings

## Overview

QA agents provide machine-readable feedback through JSON-formatted findings with file locations, severity levels, and suggested fixes. This replaced unstructured prose that led to misinterpretation.

## Data Model

```python
@dataclass
class QAFinding:
    file: str                          # Relative path from repo root
    line_number: Optional[int]         # Line where issue occurs
    severity: str                      # CRITICAL|HIGH|MAJOR|MEDIUM|LOW|MINOR|SUGGESTION
    description: str                   # What's wrong
    suggested_fix: Optional[str]       # How to fix it
    category: str                      # security|performance|correctness|readability|testing|best_practices
```

## JSON Schema

QA agents output findings in a JSON code fence:

```json
{
  "findings": [
    {
      "file": "src/handlers/auth.py",
      "line_number": 42,
      "severity": "CRITICAL",
      "category": "security",
      "description": "SQL injection in login handler",
      "suggested_fix": "Use parameterized queries"
    }
  ],
  "summary": "Found 1 critical security issue"
}
```

End with `APPROVE` or `REQUEST_CHANGES`.

## Severity Levels

| Level | Action Required |
|-------|-----------------|
| **CRITICAL** | Security vulnerability, data loss, core breakage — must fix |
| **HIGH** | Performance degradation, architectural issue — must fix |
| **MAJOR** | Significant logic error or API misuse — should fix |
| **MEDIUM** | Maintainability issue — should fix |
| **LOW/MINOR** | Style or minor inefficiency — optional |
| **SUGGESTION** | Future improvement — consider later |

## Engineer Workflow

Engineers receive fix tasks with a numbered checklist:

```markdown
### 1. CRITICAL: Security (src/handlers/auth.py:42)
**Issue**: SQL injection in login handler
**Suggested Fix**: Use parameterized queries
```

Structured findings are also available programmatically in `task.context["structured_findings"]`.

## Data Flow

```
QA Agent Response (JSON in code fence)
  → _parse_structured_findings() → QAFinding list
  → _build_review_fix_task() → Task with checklist + context
  → Engineer receives fix task
```

Parser falls back to legacy regex extraction if JSON parsing fails. Both formats coexist seamlessly.

## Code Locations

| Component | Location |
|-----------|----------|
| `QAFinding` dataclass | `core/agent.py` |
| `_parse_structured_findings()` | `core/agent.py` |
| `_format_findings_checklist()` | `core/agent.py` |
| QA prompt config | `config/agents.yaml` |
