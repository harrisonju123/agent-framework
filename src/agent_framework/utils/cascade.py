"""Generic cascade resolver for priority-ordered fallback patterns.

Many subsystems (upstream context, verdict evaluation, git diff strategies,
workflow step detection, specialization selection) use a priority cascade
where the first non-None result wins. This utility standardizes the pattern
and captures skip reasons for observability — callers decide how to log/store.
"""

from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass
class CascadeLevel(Generic[T]):
    """One level in a priority cascade.

    Attributes:
        name: Human-readable identifier for logging/metrics.
        resolve: Callable that returns (result_or_None, skip_reason).
                 None result means "fall through to next level".
                 Empty skip_reason is omitted from the skip list.
    """

    name: str
    resolve: Callable[..., tuple[Optional[T], str]]


@dataclass
class CascadeResult(Generic[T]):
    """Outcome of running a cascade.

    Attributes:
        value: The winning result, or `default` if all levels returned None.
        source: Name of the winning level, or "none" if default was used.
        skip_reasons: "{level}: {reason}" for each skipped level that
                      provided a non-empty reason.
    """

    value: T
    source: str
    skip_reasons: list[str] = field(default_factory=list)


def resolve_cascade(
    levels: Sequence[CascadeLevel[T]],
    default: T,
    *args,
    **kwargs,
) -> CascadeResult[T]:
    """Run levels in priority order, return first non-None result.

    Each level's ``resolve`` callable is invoked with ``(*args, **kwargs)``.
    The first to return a non-None first element wins. Levels that return
    None contribute their skip reason (if non-empty) to the result.

    Pure function — no side effects, no logging.
    """
    skip_reasons: list[str] = []

    for level in levels:
        result, reason = level.resolve(*args, **kwargs)
        if result is not None:
            return CascadeResult(value=result, source=level.name, skip_reasons=skip_reasons)
        if reason:
            skip_reasons.append(f"{level.name}: {reason}")

    return CascadeResult(value=default, source="none", skip_reasons=skip_reasons)
