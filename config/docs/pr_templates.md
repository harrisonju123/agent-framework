# Pull Request Description Templates

**Version:** 1.0
**Last Updated:** 2026-02-04
**Status:** Active

## Overview

Standardized PR descriptions improve code review efficiency and provide clear documentation of changes.

## Base Template

All PRs should follow this structure:

```markdown
## Summary
Implements {jira_link_or_task_id} - {title}

{Brief 2-3 sentence description of what changed and why}

## Changes
{git diff --stat output showing files changed}

## Testing
- ✓ {count} tests passed
- Test files: {list test files}
- Coverage: {coverage percentage if available}

## Acceptance Criteria
{Checklist from task.acceptance_criteria}
- [x] Criterion 1 - verified by test_foo()
- [x] Criterion 2 - verified by test_bar()
- [x] Criterion 3 - manual verification

## Change Metrics
- 📊 {line_count} lines changed across {file_count} files
- ✓ Size: {Small/Medium/Large}

{Optional sections below}
```

## JIRA Link Format

### With JIRA Integration

```markdown
Implements [PROJ-123](https://jira.company.com/browse/PROJ-123) - Add user authentication
```

### Without JIRA (Local Mode)

```markdown
Implements task impl-auth-20260204 - Add user authentication
```

## Change Size Categories

| Category | Lines | Files | Review Time | Label |
|----------|-------|-------|-------------|-------|
| Small | <100 | 1-3 | <15 min | ✓ Easy to review |
| Medium | 100-300 | 4-8 | 15-30 min | ⚠️ Reviewable but substantial |
| Large | 300-500 | 9-15 | 30-60 min | ⚠️ Consider splitting |
| Too Large | >500 | >15 | >60 min | ❌ Must split |

## Agent-Specific Additions

### Engineer PR (Simple Workflow)

```markdown
## Summary
Implements [PROJ-123](https://jira.company.com/browse/PROJ-123) - Add user authentication

Added JWT-based authentication to API endpoints.

## Changes
 src/api/auth.ts        | 145 ++++++++++++++++++++++++
 src/middleware/jwt.ts  |  89 +++++++++++++++
 src/api/auth.test.ts   | 234 +++++++++++++++++++++++++++++++++++++
 3 files changed, 468 insertions(+)

## Testing
- ✓ 34 tests passed
- Test files: src/api/auth.test.ts
- All edge cases covered (invalid tokens, expired tokens, missing tokens)

## Acceptance Criteria
- [x] POST /api/auth/login accepts email and password - verified by test_login_success()
- [x] Returns JWT token on valid credentials - verified by test_token_generation()
- [x] Returns 401 on invalid credentials - verified by test_invalid_credentials()
- [x] Rate limiting enforced (5 attempts/minute) - verified by test_rate_limiting()

## Change Metrics
- 📊 468 lines changed across 3 files
- ✓ Size: Large (reviewable but substantial)

## Implementation Notes
- Used jsonwebtoken library for JWT generation
- Followed existing pattern from src/api/users.ts
- Rate limiting middleware reused from existing endpoints
```

### QA PR (Standard Workflow)

```markdown
## Summary
Implements [PROJ-456](https://jira.company.com/browse/PROJ-456) - Fix payment processing bug

Fixed race condition in payment processor that caused duplicate charges.

## Changes
 src/payments/processor.ts      | 23 +++++++----
 src/payments/processor.test.ts | 45 +++++++++++++++++++
 2 files changed, 61 insertions(+), 7 deletions(-)

## Testing
- ✓ 12 tests passed (8 existing + 4 new)
- Test files: src/payments/processor.test.ts
- Verified fix with concurrent request simulation

## QA Verification
✅ **All acceptance criteria verified**

- [x] No duplicate charges under concurrent requests - verified by test_concurrent_payments()
- [x] Transaction idempotency maintained - verified by test_idempotent_processing()
- [x] Error handling unchanged - verified by test_error_scenarios()

### Static Analysis
```
✓ golangci-lint passed (0 issues)
✓ All tests passing
✓ No new security warnings
```

## Acceptance Criteria
- [x] Duplicate charges prevented - verified by test_concurrent_payments()
- [x] Existing functionality preserved - verified by full test suite
- [x] Transaction logs accurate - manually verified in staging

## Change Metrics
- 📊 68 lines changed across 2 files
- ✓ Size: Small (easy to review)
```

### Architect PR (Full Workflow)

```markdown
## Summary
Implements [EPIC-789](https://jira.company.com/browse/EPIC-789) - Real-time notification system

Added WebSocket-based real-time notifications with Redis pub/sub backend.

## Changes
 src/notifications/websocket.ts     | 234 ++++++++++++++++++++++++++
 src/notifications/redis-client.ts  | 156 ++++++++++++++++++
 src/notifications/types.ts         |  45 ++++++
 src/api/notifications.ts           |  89 ++++++++++
 config/redis.yaml                  |  12 ++
 docker-compose.yaml                |  15 ++
 tests/notifications/websocket.test.ts | 445 +++++++++++++++++++++++++++++++++++++++++++
 7 files changed, 996 insertions(+)

## Architecture Review
✅ **Implementation follows architectural plan**

- Connection pooling implemented as specified
- Redis pub/sub pattern matches design doc
- Event types match schema in ADR-012
- Error handling follows retry policy
- Monitoring hooks added for observability

## Testing
- ✓ 67 tests passed (67 new)
- Test files: tests/notifications/websocket.test.ts
- Coverage: 94% (above 80% threshold)
- Load tested: 10k concurrent connections

## QA Verification
✅ **QA approved** (see task qa-verification-epic789)

- All acceptance criteria verified
- Integration tests passing
- Performance benchmarks met

## Acceptance Criteria
- [x] WebSocket connections stable - verified by load tests
- [x] Message delivery guaranteed - verified by test_message_delivery()
- [x] Reconnection handling - verified by test_reconnection()
- [x] Redis failover handled - verified by test_redis_failover()
- [x] <100ms latency p95 - verified by performance tests

## Change Metrics
- 📊 996 lines changed across 7 files
- ⚠️ Size: Large (but cohesive feature)
- Reviewed in architectural context - not split further

## Deployment Notes
- Requires Redis instance (see docker-compose.yaml)
- Environment variables: REDIS_URL, REDIS_PASSWORD
- Backward compatible - can enable per-user with feature flag
```

## Optional Sections

Add these sections when relevant:

### Breaking Changes

```markdown
## ⚠️ Breaking Changes
- Removed deprecated `legacy_auth()` function
- Updated API response format (v2 → v3)
- Migration guide: docs/migration-v3.md
```

### Database Migrations

```markdown
## Database Migrations
- Migration: `20260204_add_user_roles.sql`
- Run before deployment: `npm run migrate`
- Rollback: `npm run migrate:rollback`
```

### Configuration Changes

```markdown
## Configuration
New environment variables required:
- `JWT_SECRET` - JWT signing secret
- `JWT_EXPIRY` - Token expiration (default: 24h)
```

### Dependencies

```markdown
## Dependencies
New packages added:
- `jsonwebtoken@9.0.0` - JWT generation
- `bcrypt@5.1.0` - Password hashing
```

## PR Title Format

### Standard Format

```
{type}: {brief description} [JIRA-KEY]
```

Examples:
- `feat: Add user authentication [PROJ-123]`
- `fix: Prevent duplicate payments [PROJ-456]`
- `refactor: Extract notification service [EPIC-789]`

### Types

- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code refactoring
- `perf` - Performance improvement
- `test` - Test additions/fixes
- `docs` - Documentation
- `chore` - Maintenance

### Without JIRA

```
{type}: {brief description} (task-{id})
```

Example: `feat: Add user authentication (task-impl-auth-20260204)`

## Best Practices

1. **Keep summaries concise** - 2-3 sentences max
2. **Link to JIRA** for full context (when available)
3. **Include git diff --stat** for quick file overview
4. **Map acceptance criteria to tests** - reviewers can verify
5. **Add change metrics** - helps reviewers estimate effort
6. **Document breaking changes** prominently
7. **Include deployment notes** if configuration changes needed
8. **Reference related PRs** if part of larger initiative
