# Structured QA Findings

## Overview

The structured findings feature enables QA agents to provide detailed, machine-readable feedback to engineers through JSON-formatted findings with file locations, severity levels, and suggested fixes.

## Background

**Problem**: Previously, QA findings were extracted via regex and forwarded as truncated text (max 500 chars or severity-tagged lines). Engineers received unstructured prose, leading to misinterpretation or missed issues.

**Solution**: QA agents now produce structured JSON findings with:
- Exact file and line number references
- Severity classification (CRITICAL to SUGGESTION)
- Category tags (security, performance, correctness, etc.)
- Specific suggested fixes
- Numbered checklists for systematic resolution

## Data Model

### QAFinding

The core structured finding object with the following fields:

```python
@dataclass
class QAFinding:
    file: str                          # Relative path from repo root
    line_number: Optional[int]         # Line number where issue occurs
    severity: str                      # CRITICAL|HIGH|MAJOR|MEDIUM|LOW|MINOR|SUGGESTION
    description: str                   # What's wrong
    suggested_fix: Optional[str]       # How to fix it
    category: str                      # security|performance|correctness|readability|testing|best_practices
```

**Example**:
```python
QAFinding(
    file="src/handlers/auth.py",
    line_number=42,
    severity="CRITICAL",
    description="SQL injection vulnerability in login handler",
    suggested_fix="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE email = ?', (email,))",
    category="security"
)
```

### ReviewOutcome

The parsed result of a QA review, containing structured findings:

```python
@dataclass
class ReviewOutcome:
    approved: bool                     # PR approved or changes requested
    has_critical_issues: bool          # Contains CRITICAL severity findings
    has_test_failures: bool            # Tests failed during review
    has_change_requests: bool          # Review requested changes
    has_major_issues: bool             # Contains MAJOR severity findings
    findings_summary: str              # High-level overview
    structured_findings: List[QAFinding]  # Parsed findings (if any)
```

## JSON Schema

### QA Agent Output Format

When performing code review, QA agents output findings in a JSON code fence:

```markdown
Review complete. Found 2 issues:

```json
{
  "findings": [
    {
      "file": "src/handlers/auth.py",
      "line_number": 42,
      "severity": "CRITICAL",
      "category": "security",
      "description": "SQL injection in login handler",
      "suggested_fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE email = ?', (email,))"
    },
    {
      "file": "src/api/users.py",
      "line_number": 87,
      "severity": "HIGH",
      "category": "performance",
      "description": "N+1 query in user list endpoint",
      "suggested_fix": "Use select_related() to fetch related profiles"
    }
  ],
  "summary": "Found 1 critical security issue and 1 high-priority performance issue"
}
```

Please address these issues before merging.

Overall: **REQUEST_CHANGES**
```

**Key points**:
- Place JSON in a code fence (```json...```)
- Include file:line when possible
- Provide specific suggested fixes
- End with APPROVE or REQUEST_CHANGES

## Severity Levels

Findings use a 7-level severity scale, ordered by criticality:

| Level | Meaning | Action Required |
|-------|---------|-----------------|
| **CRITICAL** | Security vulnerability, data loss risk, or core functionality broken | Must fix before merge |
| **HIGH** | Performance degradation, architectural issue, or major correctness problem | Must fix before merge |
| **MAJOR** | Significant logic error or API misuse | Should fix before merge |
| **MEDIUM** | Code quality issue affecting maintainability | Should fix before merge |
| **LOW** | Minor inefficiency or style issue | Nice to fix |
| **MINOR** | Nitpick or very minor improvement | Optional |
| **SUGGESTION** | Recommendation for future improvement | Consider for next iteration |

## Categories

Findings are categorized by type for easy filtering:

| Category | Examples |
|----------|----------|
| **security** | SQL injection, XSS, authentication bypass, secrets in code |
| **performance** | N+1 queries, unbounded loops, missing indexes, memory leaks |
| **correctness** | Logic errors, off-by-one bugs, incorrect return values, edge cases |
| **readability** | Unclear naming, duplicate code, overly complex logic |
| **testing** | Missing tests, untested edge cases, flaky tests |
| **best_practices** | Error handling, resource management, API conventions |

## Engineer Workflow

### Receiving Fix Tasks

When QA requests changes, you'll receive a fix task with a numbered checklist:

```markdown
QA review found 2 issue(s) that need fixing.

## Summary
Found 1 critical security issue and 1 high-priority performance issue

## Issues to Address

### 1. 🔴 CRITICAL: Security (src/handlers/auth.py:42)
**Issue**: SQL injection in login handler
**Suggested Fix**: Use parameterized queries: cursor.execute('SELECT * FROM users WHERE email = ?', (email,))

### 2. 🟠 HIGH: Performance (src/api/users.py:87)
**Issue**: N+1 query in user list endpoint
**Suggested Fix**: Use select_related() to fetch related profiles

## Instructions
1. Review each finding above
2. Fix the issues in the specified files/lines
3. Run tests to verify fixes
4. Commit and push your changes
5. The review will be automatically re-queued
```

### Programmatic Access

Structured findings are available in `task.context["structured_findings"]` when processing fix tasks:

```python
if "structured_findings" in task.context:
    findings_data = task.context["structured_findings"]

    for finding in findings_data["findings"]:
        file_path = finding.get("file")
        line_num = finding.get("line_number")
        severity = finding["severity"]
        description = finding["description"]
        suggested_fix = finding.get("suggested_fix")
        category = finding.get("category")

        print(f"{severity}: {file_path}:{line_num} - {description}")
        if suggested_fix:
            print(f"  Fix: {suggested_fix}")
        # Apply fix...
```

## QA Agent Prompts

The QA agent is configured to produce structured findings. The relevant configuration can be found in `config/agents.yaml`.

**Key prompt directives**:
1. Output findings in JSON code blocks
2. Include exact file:line references
3. Provide specific, actionable suggested fixes
4. Classify severity accurately
5. Assign appropriate categories
6. Conclude with APPROVE or REQUEST_CHANGES

The QA prompt instructs the agent to structure feedback as JSON before processing, ensuring consistency and machine-readability.

## Data Flow

```
QA Agent Review
    ↓
LLM Response (JSON in code fence)
    ↓
_parse_structured_findings() → QAFinding list | None
    ↓
_extract_review_findings() → JSON string | text fallback
    ↓
_build_review_fix_task() → Task with checklist + context
    ↓
Engineer receives fix task with numbered checklist
```

### Parsing Process

**Location**: `src/agent_framework/core/agent.py:_parse_structured_findings()`

The parser:
1. Searches for JSON code blocks in QA response
2. Extracts and validates JSON
3. Creates QAFinding objects from each finding
4. Returns list of QAFinding or None if parsing fails
5. Automatically falls back to legacy regex extraction on failure

### Formatting Process

**Location**: `src/agent_framework/core/agent.py:_format_findings_checklist()`

The formatter converts structured findings into an engineer-friendly numbered checklist with:
- Severity emoji indicators (🔴 CRITICAL, 🟠 HIGH, 🟡 MAJOR, etc.)
- File:line references with hyperlinks to source
- Description and suggested fix on separate lines
- Checkbox items [ ] for tracking completion

### Task Building

**Location**: `src/agent_framework/core/agent.py:_build_review_fix_task()`

When QA requests changes, a fix task is created with:
- Title: "Address code review feedback for {jira_key}"
- Description: Formatted checklist of findings
- Context: Full structured findings JSON for programmatic access
- Type: "fix"
- Depends on: Empty (can be picked up immediately)

## Backward Compatibility

The system gracefully falls back to legacy format when needed:

1. **QA outputs text only**: Parser returns `None`, legacy regex extraction runs
2. **Malformed JSON**: Try/except catches errors, uses text fallback
3. **Engineer receives legacy task**: Description shows text findings (no checklist)

**No breaking changes** - old and new formats coexist seamlessly. Engineers can work with either format, though structured findings provide better experience.

## Code Locations

All structured findings code is in `src/agent_framework/core/agent.py`:

| Component | Function | Purpose |
|-----------|----------|---------|
| **Models** | `QAFinding` dataclass | Represents a single finding |
| | `ReviewOutcome` dataclass | Contains findings + review result |
| **Parser** | `_parse_structured_findings()` | Extracts JSON from QA response |
| **Formatter** | `_format_findings_checklist()` | Creates numbered checklist |
| **Task Builder** | `_build_review_fix_task()` | Generates fix task with findings |

Configuration is in `config/agents.yaml` (qa agent prompt section).

## Testing

Run the structured findings test suite:

```bash
# Unit tests for structured findings parsing and formatting
pytest tests/unit/test_structured_findings.py -v

# Integration tests for review workflow with fix tasks
pytest tests/unit/test_review_fix.py -v

# Full test suite
pytest tests/unit/ -v
```

### Test Coverage

Tests verify:
- ✅ JSON parsing of valid findings
- ✅ Handling of malformed JSON (fallback)
- ✅ QAFinding dataclass validation
- ✅ Checklist formatting with proper emoji/styling
- ✅ Context passing to fix tasks
- ✅ Backward compatibility with text findings
- ✅ Line number handling (optional field)
- ✅ Severity and category classification

## Troubleshooting

### QA not producing JSON

**Symptom**: Fix tasks show text findings, not checklists

**Causes**:
1. QA agent prompt doesn't include structured findings section
2. LLM chose not to use JSON format
3. QA disabled or misconfigured

**Steps to Fix**:
1. Check `config/agents.yaml` - QA prompt includes structured findings section?
2. Review QA response - does it contain JSON in code block?
3. Check logs: `grep "parsing findings" logs/agent.log`
4. Verify QA agent is enabled: `grep "qa:" config/agents.yaml`

### Malformed JSON in findings

**Symptom**: Logs show `Failed to parse code fence JSON findings` warnings

**Cause**: LLM produced syntactically invalid JSON (missing commas, quotes, etc.)

**Steps to Fix**:
1. Review QA response for JSON syntax errors
2. System automatically falls back to text format
3. No action needed - engineer still gets findings
4. If frequent, consider adjusting QA prompt for stricter JSON validation

### Missing file:line references

**Symptom**: Checklist shows `(issue in file.py)` without line number

**Cause**: QA didn't specify line number in finding

**Why It's OK**: Line number is optional field - system still works

**Improvement**:
1. Review QA prompt to emphasize importance of line numbers
2. Add examples with line numbers to QA prompt
3. Monitor findings over time - should improve with iterations

### Findings not appearing in fix task context

**Symptom**: `task.context["structured_findings"]` is empty or missing

**Cause**:
1. QA provided text findings (fallback path)
2. JSON parsing failed silently

**Debug**:
1. Check task.context for "findings_summary" (text fallback indicator)
2. Review agent logs for parse errors
3. Manually verify QA response contains valid JSON

### Engineer receiving old-style tasks

**Symptom**: Fix tasks show text findings instead of numbered checklist

**Cause**:
1. QA agent not using structured findings feature yet
2. Migration in progress
3. Feature disabled

**Solution**:
1. Ensure QA agent is updated with structured findings prompt
2. Verify config/agents.yaml has updated QA section
3. Restart QA agent to pick up config changes

## Migration Guide

If you have an existing code review workflow:

### Phase 1: Update QA Agent
1. Update `config/agents.yaml` QA prompt to include structured findings section
2. Add examples of JSON output format
3. Restart QA agent
4. Monitor logs for successful parsing

### Phase 2: Test with New Tasks
1. Create a test review task
2. Verify QA outputs JSON findings
3. Confirm fix task shows numbered checklist
4. Test programmatic access to context data

### Phase 3: Deprecate Legacy Format (Optional)
Once confident with structured findings:
1. Remove legacy regex extraction code (optional)
2. Update team docs to use new format
3. Archive old tasks for reference

**No forced migration needed** - both formats work together.

## Performance Considerations

Structured findings have minimal performance impact:

- **Parsing**: ~1ms for typical review (few findings)
- **Storage**: JSON adds ~10% to task file size (negligible)
- **Processing**: Same LLM tokens (inherent to review task)

No special optimization needed for typical workloads (< 1000 tasks/day).

## Future Enhancements

Potential improvements to the structured findings system:

1. **Finding Resolution Tracking**: Mark findings as resolved in JIRA
   - Engineers can update resolution status
   - Metrics tracking (% critical resolved, etc.)

2. **Analytics Dashboard**: Most common finding types
   - Heatmap of files with most issues
   - Severity trends over time
   - Category distribution

3. **Auto-Fix Suggestions**: Generate code diffs from findings
   - For simple fixes (imports, formatting)
   - Reduces engineer friction

4. **Severity-Based Routing**: Auto-escalate critical findings
   - CRITICAL → Architect review
   - HIGH + complex → Senior engineer
   - Others → Assigned engineer

5. **Finding Templates**: Pre-defined patterns
   - SQL injection template
   - N+1 query template
   - Race condition template
   - Consistency across teams

6. **Integration with IDEs**: Jump-to-line in VS Code/IntelliJ
   - Click finding → Open file at line in IDE
   - Quick apply suggested fixes

## See Also

- [Code Review Workflow](CODE_REVIEW_WORKFLOW.md) - Complete review process including structured findings
- [Task Model](../src/agent_framework/core/task.py) - Task data structures
- [Agent Configuration](../config/agents.yaml) - Agent prompts and settings
- [Review Fixes](REVIEW_FIXES.md) - Engineer workflow for addressing findings
