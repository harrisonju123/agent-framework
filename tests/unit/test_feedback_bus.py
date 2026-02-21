"""Tests for FeedbackBus cross-feature learning signals."""

import pytest

from agent_framework.core.feedback_bus import FeedbackBus, QA_WARNINGS_MAX_CHARS
from agent_framework.memory.memory_store import MemoryStore


REPO = "owner/repo"
AGENT = "engineer"


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def bus(store):
    return FeedbackBus(store)


@pytest.fixture
def disabled_bus(tmp_path):
    return FeedbackBus(MemoryStore(workspace=tmp_path, enabled=False))


# --- FeedbackBus.enabled ---


class TestEnabled:
    def test_enabled_when_store_enabled(self, bus):
        assert bus.enabled is True

    def test_disabled_when_store_disabled(self, disabled_bus):
        assert disabled_bus.enabled is False


# --- store_self_eval_failure ---


class TestStoreSelfEvalFailure:
    def test_returns_zero_when_disabled(self, disabled_bus):
        count = disabled_bus.store_self_eval_failure(
            REPO, AGENT, "task-1", "FAIL: missing tests",
            acceptance_criteria=["All tests pass"],
        )
        assert count == 0

    def test_stores_matched_criteria(self, bus, store):
        # Critique must share enough keywords (2+) with a criterion to trigger Strategy 1
        critique = "FAIL: The tests are failing and coverage is below threshold"
        criteria = [
            "All tests pass with coverage above 80%",
            "API endpoint returns 200",
        ]
        count = bus.store_self_eval_failure(
            REPO, AGENT, "task-1", critique, acceptance_criteria=criteria,
        )
        assert count >= 1

        memories = store.recall(REPO, AGENT, category="missed_criteria")
        assert len(memories) >= 1
        # Strategy 1 produces "Commonly missed:", Strategy 2 produces "Self-eval gap:"
        assert any(
            "Commonly missed:" in m.content or "Self-eval gap:" in m.content
            for m in memories
        )
        assert any("self_eval" in m.tags for m in memories)

    def test_extracts_gaps_from_critique_text(self, bus, store):
        critique = (
            "FAIL: missing implementation of the logging module. "
            "The code lacks error handling for edge cases."
        )
        count = bus.store_self_eval_failure(
            REPO, AGENT, "task-2", critique, acceptance_criteria=None,
        )
        assert count >= 1

        memories = store.recall(REPO, AGENT, category="missed_criteria")
        assert any("Self-eval gap:" in m.content for m in memories)

    def test_caps_at_three_extracted_gaps(self, bus, store):
        critique = (
            "FAIL: missing feature A implementation completely. "
            "The code lacks feature B error handling. "
            "No feature C authentication support. "
            "Missing feature D logging integration. "
            "Without feature E caching mechanism. "
        )
        count = bus.store_self_eval_failure(
            REPO, AGENT, "task-3", critique, acceptance_criteria=None,
        )
        assert count <= 3

    def test_no_criteria_no_gaps_returns_zero(self, bus):
        count = bus.store_self_eval_failure(
            REPO, AGENT, "task-4", "PASS", acceptance_criteria=None,
        )
        assert count == 0


# --- store_qa_pattern ---


class TestStoreQaPattern:
    def test_returns_zero_when_disabled(self, disabled_bus):
        count = disabled_bus.store_qa_pattern(
            REPO, AGENT, "task-1", {"findings": [{"description": "XSS issue"}]},
        )
        assert count == 0

    def test_returns_zero_for_empty_findings(self, bus):
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", {"findings": []})
        assert count == 0

    def test_returns_zero_for_missing_findings_key(self, bus):
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", {})
        assert count == 0

    def test_stores_recurring_findings(self, bus, store):
        findings = {
            "findings": [
                {"description": "SQL injection in user query", "severity": "CRITICAL", "file": "api/users.py"},
                {"description": "SQL injection in order query", "severity": "HIGH", "file": "api/orders.py"},
            ],
        }
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", findings)
        assert count >= 1

        memories = store.recall(REPO, AGENT, category="qa_patterns")
        assert len(memories) >= 1
        assert any("security" in m.tags for m in memories)

    def test_stores_critical_single_findings(self, bus, store):
        findings = {
            "findings": [
                {"description": "SQL injection in authentication", "severity": "CRITICAL", "file": "auth.py"},
            ],
        }
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", findings)
        assert count >= 1

    def test_skips_non_recurring_non_critical(self, bus, store):
        findings = {
            "findings": [
                {"description": "Minor formatting issue", "severity": "LOW", "file": "utils.py"},
            ],
        }
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", findings)
        assert count == 0

    def test_classifies_multiple_domains(self, bus, store):
        findings = {
            "findings": [
                {"description": "Slow database query performance", "severity": "HIGH", "file": "db.py"},
                {"description": "Memory leak in cache performance issue", "severity": "HIGH", "file": "cache.py"},
                {"description": "Missing test coverage for handler", "severity": "MEDIUM", "file": "test_handler.py"},
                {"description": "No assertions in test file", "severity": "MEDIUM", "file": "test_api.py"},
            ],
        }
        count = bus.store_qa_pattern(REPO, AGENT, "task-1", findings)
        assert count >= 2

        memories = store.recall(REPO, AGENT, category="qa_patterns")
        domains = set()
        for m in memories:
            domains.update(m.tags)
        assert "performance" in domains
        assert "testing" in domains


# --- store_specialization_signal ---


class TestStoreSpecializationSignal:
    def test_returns_false_when_disabled(self, disabled_bus):
        result = disabled_bus.store_specialization_signal(
            REPO, "task-1",
            topic="Should we use React or Vue for the frontend UI component design?",
            recommendation="Use React for the frontend component architecture.",
        )
        assert result is False

    def test_stores_hint_when_domain_detected(self, bus, store):
        result = bus.store_specialization_signal(
            REPO, "task-1",
            topic="Should we use React or Vue for the frontend UI component design?",
            recommendation="Use React for the frontend component architecture.",
        )
        assert result is True

        memories = store.recall(REPO, "shared", category="specialization_hints")
        assert len(memories) == 1
        assert "frontend" in memories[0].content
        assert "debate" in memories[0].tags

    def test_skips_when_current_matches(self, bus):
        result = bus.store_specialization_signal(
            REPO, "task-1",
            topic="Should we use React or Vue for the frontend UI component design?",
            recommendation="Use React for the frontend component browser integration.",
            current_specialization="frontend",
        )
        assert result is False

    def test_skips_when_no_domain_keywords(self, bus):
        result = bus.store_specialization_signal(
            REPO, "task-1",
            topic="Should we add logging?",
            recommendation="Yes, add structured logging.",
        )
        assert result is False

    def test_detects_backend_domain(self, bus, store):
        result = bus.store_specialization_signal(
            REPO, "task-1",
            topic="Which database query pattern for the API endpoint?",
            recommendation="Use prepared statements for the database API.",
        )
        assert result is True

        memories = store.recall(REPO, "shared", category="specialization_hints")
        assert len(memories) == 1
        assert "backend" in memories[0].tags


# --- get_qa_warnings ---


class TestGetQaWarnings:
    def test_empty_when_no_patterns(self, bus):
        result = bus.get_qa_warnings(REPO, AGENT)
        assert result == ""

    def test_returns_formatted_warnings(self, bus, store):
        # Pre-seed some qa_patterns memories
        store.remember(REPO, AGENT, "qa_patterns", "QA pattern (security): 3 findings — [HIGH] SQL injection in users", tags=["qa", "security"])
        store.remember(REPO, AGENT, "missed_criteria", "Commonly missed: All tests pass", tags=["self_eval"])

        result = bus.get_qa_warnings(REPO, AGENT)
        assert "## QA Pattern Warnings" in result
        assert "recurring issues" in result
        assert "security" in result
        assert "tests pass" in result

    def test_respects_max_chars(self, bus, store):
        # Seed many patterns
        for i in range(20):
            store.remember(
                REPO, AGENT, "qa_patterns",
                f"QA pattern (domain_{i}): many findings — some very long description that takes space #{i}",
                tags=["qa"],
            )

        result = bus.get_qa_warnings(REPO, AGENT, max_chars=QA_WARNINGS_MAX_CHARS)
        assert len(result) <= QA_WARNINGS_MAX_CHARS + 50  # small tolerance for trailing newline


# --- get_specialization_hints ---


class TestGetSpecializationHints:
    def test_empty_when_no_hints(self, bus):
        result = bus.get_specialization_hints(REPO)
        assert result == []

    def test_returns_hints(self, bus, store):
        store.remember(REPO, "shared", "specialization_hints", "Debate suggested 'backend'", tags=["debate", "backend"])

        result = bus.get_specialization_hints(REPO)
        assert len(result) == 1
        assert "backend" in result[0].content


# --- _classify_finding ---


class TestClassifyFinding:
    @pytest.mark.parametrize("desc,file,expected", [
        ("SQL injection vulnerability", "api.py", "security"),
        ("XSS in user input", "form.js", "security"),
        ("N+1 query in user list", "users.py", "performance"),
        ("Slow cache invalidation", "cache.py", "performance"),
        ("Missing test coverage", "handler.py", "testing"),
        ("No assertions found", "test_api.py", "testing"),
        ("Unhandled exception in handler", "handler.py", "error_handling"),
        ("Missing type annotations", "models.py", "type_safety"),
        ("No logging in critical path", "service.py", "observability"),
        ("Variable naming inconsistency", "utils.py", "code_quality"),
    ])
    def test_classification(self, desc, file, expected):
        assert FeedbackBus._classify_finding(desc, file) == expected


# --- _detect_domains ---


class TestDetectDomains:
    def test_detects_frontend(self):
        domains = FeedbackBus._detect_domains("The React component UI needs updating")
        assert "frontend" in domains

    def test_detects_backend(self):
        domains = FeedbackBus._detect_domains("The API endpoint needs a new database query")
        assert "backend" in domains

    def test_detects_infrastructure(self):
        domains = FeedbackBus._detect_domains("Configure Docker and Kubernetes for deploy")
        assert "infrastructure" in domains

    def test_requires_two_keywords(self):
        # Single keyword shouldn't trigger
        domains = FeedbackBus._detect_domains("The API is slow")
        assert "backend" not in domains

    def test_no_match(self):
        domains = FeedbackBus._detect_domains("General code improvement needed")
        assert len(domains) == 0
