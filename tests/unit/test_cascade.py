"""Tests for the generic cascade resolver utility."""

from typing import Optional

from agent_framework.utils.cascade import CascadeLevel, CascadeResult, resolve_cascade


def _level(name: str, value: Optional[str], reason: str = ""):
    """Helper: build a CascadeLevel that returns a fixed (value, reason)."""
    return CascadeLevel(name=name, resolve=lambda **kw: (value, reason))


class TestResolveCascade:
    """Core behavior of resolve_cascade()."""

    def test_first_level_wins(self):
        levels = [
            _level("alpha", "ALPHA"),
            _level("beta", "BETA"),
        ]
        result = resolve_cascade(levels, default="DEFAULT")
        assert result.value == "ALPHA"
        assert result.source == "alpha"
        assert result.skip_reasons == []

    def test_second_level_wins_after_first_skipped(self):
        levels = [
            _level("alpha", None, "not applicable"),
            _level("beta", "BETA"),
        ]
        result = resolve_cascade(levels, default="DEFAULT")
        assert result.value == "BETA"
        assert result.source == "beta"
        assert result.skip_reasons == ["alpha: not applicable"]

    def test_third_level_wins(self):
        levels = [
            _level("alpha", None, "missing"),
            _level("beta", None, "empty"),
            _level("gamma", "GAMMA"),
        ]
        result = resolve_cascade(levels, default="DEFAULT")
        assert result.value == "GAMMA"
        assert result.source == "gamma"
        assert result.skip_reasons == ["alpha: missing", "beta: empty"]

    def test_default_when_all_none(self):
        levels = [
            _level("alpha", None, "no data"),
            _level("beta", None, "no data"),
        ]
        result = resolve_cascade(levels, default="FALLBACK")
        assert result.value == "FALLBACK"
        assert result.source == "none"
        assert len(result.skip_reasons) == 2

    def test_empty_skip_reason_omitted(self):
        """Levels with empty skip reasons don't pollute the list."""
        levels = [
            _level("alpha", None, ""),
            _level("beta", None, "has reason"),
            _level("gamma", "WIN"),
        ]
        result = resolve_cascade(levels, default="DEFAULT")
        assert result.skip_reasons == ["beta: has reason"]

    def test_empty_levels_returns_default(self):
        result = resolve_cascade([], default="DEFAULT")
        assert result.value == "DEFAULT"
        assert result.source == "none"
        assert result.skip_reasons == []


class TestResolveCascadeKwargs:
    """Kwargs are forwarded to each resolver."""

    def test_kwargs_forwarded(self):
        def resolver(task=None):
            if task == "magic":
                return ("FOUND", "")
            return (None, "wrong task")

        levels = [CascadeLevel("finder", resolver)]
        result = resolve_cascade(levels, default="DEFAULT", task="magic")
        assert result.value == "FOUND"
        assert result.source == "finder"

    def test_kwargs_forwarded_to_all_levels(self):
        """All levels receive the same kwargs."""
        calls = []

        def recorder(name):
            def fn(**kwargs):
                calls.append((name, kwargs))
                return (None, "")
            return fn

        levels = [
            CascadeLevel("a", recorder("a")),
            CascadeLevel("b", recorder("b")),
        ]
        resolve_cascade(levels, default="", x=42, y="hello")
        assert len(calls) == 2
        assert calls[0] == ("a", {"x": 42, "y": "hello"})
        assert calls[1] == ("b", {"x": 42, "y": "hello"})


class TestCascadeResultDataclass:
    """CascadeResult is a plain dataclass."""

    def test_skip_reasons_default_empty(self):
        r = CascadeResult(value="x", source="test")
        assert r.skip_reasons == []

    def test_fields_accessible(self):
        r = CascadeResult(value="val", source="src", skip_reasons=["a: b"])
        assert r.value == "val"
        assert r.source == "src"
        assert r.skip_reasons == ["a: b"]
