# Agent Framework - Claude Code Guidelines

## Code Quality

Write clean code that works. Prioritize readability and maintainability over cleverness.

## Comments & Documentation

**Document the WHY, not the WHAT.**

Bad:
```python
# Increment counter by 1
counter += 1
```

Good:
```python
# Retry count tracks failed attempts for exponential backoff
counter += 1
```

No AI slop. Comments should add context that isn't obvious from the code itself. If the code is self-explanatory, don't comment it.

## Security

**Never commit:**
- API keys, tokens, or credentials
- Passwords or secrets (even "test" ones)
- PII (names, emails, SSNs, etc.)
- Internal URLs or IP addresses
- Database connection strings with credentials

Use environment variables or config files (gitignored) for sensitive data.

## Code Evolution

**Prefer enhancement over addition.**

Before writing new code:
1. Search for existing utilities in `utils/`
2. Check if a similar pattern exists elsewhere
3. Consider if an existing function can be extended

**Addition means deletion.**

When adding new functionality, ask: "What can I remove now?"
- New utility? Delete duplicated inline code
- New abstraction? Remove the concrete implementations it replaces
- New feature? Remove deprecated code paths

## Existing Patterns

Use established utilities:
- `utils/subprocess_utils.py` - Command execution
- `utils/error_handling.py` - Exception patterns
- `utils/validators.py` - Input validation
- `utils/atomic_io.py` - File operations

Reference config docs in `config/docs/` rather than duplicating documentation in code.

## Testing

Write tests for new functionality. Run existing tests before committing:
```bash
pytest tests/ -v
```

## Commits

Keep commits focused. One logical change per commit. Write commit messages that explain why the change was made, not just what changed.
