"""Task decomposition logic for splitting large plans into subtasks."""

import copy
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_framework.core.task import Task, PlanDocument, TaskStatus

# Action verbs that signal a discrete deliverable in approach steps
_ACTION_VERBS = re.compile(
    r"^(add|create|implement|build|write|define|set\s?up|configure|integrate|"
    r"update|modify|extend|refactor|migrate|register|wire|connect|expose|emit|"
    r"introduce|extract|generate|render|display|mount)\b",
    re.IGNORECASE,
)

# Preparatory/exploratory verbs that aren't deliverables
_PREP_VERBS = re.compile(
    r"^(read|understand|review|analyze|explore|check|examine|research|investigate|"
    r"plan|decide|consider|evaluate|study|identify|gather|document)\b",
    re.IGNORECASE,
)

# Execution-tier classification for subtask dependency inference.
# Higher tiers depend on all lower tiers; same-tier boundaries are independent.
_INFRA_DIRS = frozenset({
    "config", "setup", "infra", "infrastructure",
    "migrations", "schema", "scripts", "deploy",
})
_TEST_DIRS = frozenset({
    "tests", "test", "spec", "specs", "e2e",
    "integration", "testing", "__tests__",
})
_DEFAULT_TIER = 1


def extract_requirements_checklist(plan: PlanDocument) -> list[dict]:
    """Parse plan approach steps into a numbered checklist of discrete deliverables.

    Each checklist item represents something the engineer must ship.
    Preparatory steps (read, understand, analyze) are excluded — only
    action-oriented steps become deliverables.

    Returns list of dicts: {"id": int, "description": str, "files": list[str], "status": "pending"}
    """
    checklist = []
    item_id = 0

    for step in (plan.approach or []):
        # Strip leading numbering like "1. " or "- "
        text = re.sub(r"^\d+\.\s*", "", step).strip()
        text = re.sub(r"^[-*]\s*", "", text).strip()

        if not text:
            continue

        # Skip preparatory steps — everything else is a deliverable
        if _PREP_VERBS.match(text):
            continue

        item_id += 1
        # Find files from plan that this step mentions
        relevant_files = [
            f for f in (plan.files_to_modify or [])
            if Path(f).name.lower() in text.lower()
            or (len(Path(f).stem) > 3 and Path(f).stem.lower() in text.lower())
        ]
        checklist.append({
            "id": item_id,
            "description": text,
            "files": relevant_files,
            "status": "pending",
        })

    return checklist


def estimate_plan_lines(plan: "PlanDocument") -> int:
    """Estimate total implementation lines from plan signals.

    Takes the higher of file count (50 lines/file) vs approach step count
    (25 lines/step) to avoid double-counting overlapping signals.
    """
    file_estimate = len(plan.files_to_modify) * 50 if plan.files_to_modify else 0
    step_estimate = len(plan.approach) * 25 if plan.approach else 0
    return max(file_estimate, step_estimate)


@dataclass
class SubtaskBoundary:
    """A natural split point identified in a plan."""

    name: str  # e.g. "Database schema changes"
    files: list[str]  # Files this subtask covers (source + co-located tests)
    approach_steps: list[str]  # Relevant steps from parent plan
    depends_on_subtasks: list[int]  # Indices of subtasks this depends on (for ordering)
    estimated_lines: int
    source_file_count: int = 0  # Non-test files; used for MAX_DELIVERABLES enforcement


class TaskDecomposer:
    """Decomposes large tasks into smaller, independent subtasks."""

    # Configuration constants
    DECOMPOSE_THRESHOLD = 350  # Lines above which decomposition triggers
    TARGET_SUBTASK_SIZE = 250  # Target lines per subtask
    MIN_SUBTASK_SIZE = 50  # Don't create subtasks smaller than this
    MAX_SUBTASKS = 5  # Cap on number of subtasks
    MAX_DEPTH = 1  # Subtasks cannot themselves decompose
    MAX_DELIVERABLES_PER_SUBTASK = 4  # Source files per subtask; test files don't count

    # Refactoring existing code requires reading, understanding context, editing,
    # and fixing tests — substantially more work than greenfield creation.
    MODIFICATION_EFFORT_MULTIPLIER: float = 2.0

    # Medium tasks with many discrete deliverables should decompose even below
    # the line threshold — the deliverable count signals complexity that a single
    # context window may not reliably complete.
    REQUIREMENTS_COUNT_TRIGGER = 6
    REQUIREMENTS_MIN_LINES = 200

    def should_decompose(
        self, plan: PlanDocument, estimated_lines: int, requirements_count: int = 0
    ) -> bool:
        """
        Determine if a task should be decomposed into subtasks.

        Args:
            plan: The task's plan document
            estimated_lines: Estimated total lines of code for the task
            requirements_count: Number of discrete deliverables from checklist extraction

        Returns:
            True if task exceeds threshold and has enough files to split
        """
        # Need at least 2 files to meaningfully split
        if len(plan.files_to_modify) < 2:
            return False

        # Primary trigger: estimated lines above threshold
        if estimated_lines >= self.DECOMPOSE_THRESHOLD:
            return True

        # Secondary trigger: many discrete deliverables even if line count is moderate
        if (requirements_count >= self.REQUIREMENTS_COUNT_TRIGGER
                and estimated_lines >= self.REQUIREMENTS_MIN_LINES):
            return True

        return False

    def decompose(
        self, parent_task: Task, plan: PlanDocument, estimated_lines: int,
        workspace: str | Path | None = None,
    ) -> list[Task]:
        """
        Split parent task into independent subtasks.

        Args:
            parent_task: The task to decompose
            plan: The task's plan document
            estimated_lines: Estimated total lines of code
            workspace: Working directory for checking file existence (effort multiplier)

        Returns:
            List of created subtask objects

        Side effects:
            Updates parent_task.subtask_ids with the IDs of created subtasks
        """
        # Check if parent is already a subtask (max depth = 1)
        if parent_task.parent_task_id is not None:
            return []

        # Identify natural split boundaries
        boundaries = self._identify_split_boundaries(
            plan, estimated_lines, workspace=workspace,
        )

        # Cap at MAX_SUBTASKS
        if len(boundaries) > self.MAX_SUBTASKS:
            boundaries = boundaries[: self.MAX_SUBTASKS]

        # Create subtasks from boundaries
        subtasks = []
        for index, boundary in enumerate(boundaries):
            subtask = self._create_subtask(
                parent=parent_task,
                boundary=boundary,
                index=index + 1,
                total=len(boundaries),
                plan=plan,
            )
            subtasks.append(subtask)

        # Update parent with subtask IDs
        parent_task.subtask_ids = [task.id for task in subtasks]

        return subtasks

    def _identify_split_boundaries(
        self, plan: PlanDocument, estimated_lines: int,
        workspace: str | Path | None = None,
    ) -> list[SubtaskBoundary]:
        """
        Group files_to_modify into clusters that form natural subtasks.

        Strategy:
        1. Separate source files from test files
        2. Group source files by top-level directory
        3. Co-locate each test file with the source subtask it corresponds to
        4. Enforce MAX_DELIVERABLES_PER_SUBTASK (source files only; tests are free)
        5. Apply effort multiplier for files that already exist on disk

        Args:
            plan: The task's plan document
            estimated_lines: Total estimated lines for the task
            workspace: Working directory for checking file existence (effort multiplier)

        Returns:
            List of SubtaskBoundary objects representing natural splits
        """
        files = plan.files_to_modify
        if not files:
            return []

        # Partition into source vs test files
        source_files: list[str] = []
        test_files: list[str] = []
        for f in files:
            if self._is_test_file(f):
                test_files.append(f)
            else:
                source_files.append(f)

        # Estimate lines per source file (tests are "free" additions)
        source_count = len(source_files) or 1
        lines_per_file = estimated_lines // source_count

        # Group source files by top-level directory
        dir_groups: dict[str, list[str]] = {}
        for file in source_files:
            parts = Path(file).parts
            top_dir = parts[0] if parts else "root"
            if top_dir not in dir_groups:
                dir_groups[top_dir] = []
            dir_groups[top_dir].append(file)

        # Build initial boundaries from directory groups
        boundaries: list[SubtaskBoundary] = []
        for dir_name, group_files in dir_groups.items():
            group_estimated = len(group_files) * lines_per_file

            if group_estimated > 300 and len(group_files) > 2:
                sub_boundaries = self._split_by_subdirectory(
                    group_files, lines_per_file, dir_name
                )
                for sb in sub_boundaries:
                    sb.source_file_count = len(sb.files)
                    if not sb.approach_steps:
                        sb.approach_steps = self._extract_relevant_steps(
                            plan.approach, sb.files
                        )
                boundaries.extend(sub_boundaries)
            else:
                boundary = SubtaskBoundary(
                    name=f"Changes in {dir_name}",
                    files=group_files,
                    approach_steps=self._extract_relevant_steps(
                        plan.approach, group_files
                    ),
                    depends_on_subtasks=[],
                    estimated_lines=group_estimated,
                    source_file_count=len(group_files),
                )
                boundaries.append(boundary)

        # Enforce MAX_DELIVERABLES_PER_SUBTASK on source files
        boundaries = self._enforce_max_deliverables(boundaries, lines_per_file, plan)

        # Co-locate test files with their corresponding source boundaries
        self._colocate_test_files(boundaries, test_files)

        # Apply effort multiplier for existing files (modifications are harder)
        if workspace:
            self._apply_effort_multiplier(boundaries, workspace)

        # If we still have < 2 boundaries, fall back to approach step splitting
        if len(boundaries) < 2:
            boundaries = self._split_by_approach_steps(plan, estimated_lines)

        # Filter out boundaries that are too small
        boundaries = [b for b in boundaries if b.estimated_lines >= self.MIN_SUBTASK_SIZE]

        # Infer execution-order dependencies between surviving boundaries
        self._infer_boundary_dependencies(boundaries)

        # Ensure we have at least 2 boundaries
        if len(boundaries) < 2:
            return []

        return boundaries

    def _split_by_subdirectory(
        self, files: list[str], lines_per_file: int, parent_dir: str
    ) -> list[SubtaskBoundary]:
        """Split a large directory group by subdirectories."""
        subdir_groups: dict[str, list[str]] = {}

        for file in files:
            parts = Path(file).parts
            # Use first 2 levels (e.g., "src/core")
            subdir = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if subdir not in subdir_groups:
                subdir_groups[subdir] = []
            subdir_groups[subdir].append(file)

        boundaries = []
        for subdir_name, group_files in subdir_groups.items():
            boundary = SubtaskBoundary(
                name=f"Changes in {subdir_name}",
                files=group_files,
                approach_steps=[],  # Will be filled by caller
                depends_on_subtasks=[],
                estimated_lines=len(group_files) * lines_per_file,
                source_file_count=len(group_files),
            )
            boundaries.append(boundary)

        return boundaries

    def _split_by_approach_steps(
        self, plan: PlanDocument, estimated_lines: int
    ) -> list[SubtaskBoundary]:
        """
        Fall back to splitting by approach steps when directory grouping fails.

        Divides approach steps and files roughly in half.
        """
        # If we have less than 2 files, can't meaningfully split
        if len(plan.files_to_modify) < 2:
            return []

        mid_point = len(plan.approach) // 2
        mid_files = len(plan.files_to_modify) // 2

        boundaries = [
            SubtaskBoundary(
                name="First phase implementation",
                files=plan.files_to_modify[:mid_files],
                approach_steps=plan.approach[:mid_point],
                depends_on_subtasks=[],
                estimated_lines=estimated_lines // 2,
            ),
            SubtaskBoundary(
                name="Second phase implementation",
                files=plan.files_to_modify[mid_files:],
                approach_steps=plan.approach[mid_point:],
                depends_on_subtasks=[0],  # Second phase depends on first
                estimated_lines=estimated_lines // 2,
            ),
        ]

        return boundaries

    @staticmethod
    def _is_test_file(file_path: str) -> bool:
        """Check if a file path is a test file by directory or naming convention."""
        parts = Path(file_path).parts
        name = Path(file_path).name
        for part in parts:
            if part.lower() in _TEST_DIRS:
                return True
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        return False

    @staticmethod
    def _test_matches_source(test_path: str, source_path: str) -> bool:
        """Check if a test file corresponds to a source file.

        Matches `tests/**/test_module_name.py` to `src/.../module_name.py`.
        """
        source_stem = Path(source_path).stem
        test_name = Path(test_path).name
        return test_name == f"test_{source_stem}.py"

    def _enforce_max_deliverables(
        self,
        boundaries: list[SubtaskBoundary],
        lines_per_file: int,
        plan: PlanDocument,
    ) -> list[SubtaskBoundary]:
        """Split any boundary that exceeds MAX_DELIVERABLES_PER_SUBTASK source files.

        Chunks oversized boundaries into groups of MAX_DELIVERABLES_PER_SUBTASK files.
        """
        result: list[SubtaskBoundary] = []
        cap = self.MAX_DELIVERABLES_PER_SUBTASK

        for boundary in boundaries:
            if boundary.source_file_count <= cap:
                result.append(boundary)
                continue

            # Chunk the boundary's source files into groups of `cap`
            files = boundary.files
            for chunk_start in range(0, len(files), cap):
                chunk = files[chunk_start:chunk_start + cap]
                chunk_num = chunk_start // cap + 1
                sub = SubtaskBoundary(
                    name=f"{boundary.name} (part {chunk_num})",
                    files=chunk,
                    approach_steps=self._extract_relevant_steps(
                        plan.approach, chunk
                    ),
                    depends_on_subtasks=[],
                    estimated_lines=len(chunk) * lines_per_file,
                    source_file_count=len(chunk),
                )
                result.append(sub)

        return result

    def _colocate_test_files(
        self,
        boundaries: list[SubtaskBoundary],
        test_files: list[str],
    ) -> None:
        """Pair each test file with the boundary containing its source counterpart.

        Unmatched test files go into the last boundary as a catch-all.
        Test files don't increase source_file_count (they're "free" additions).
        """
        remaining_tests: list[str] = []

        for test_path in test_files:
            matched = False
            for boundary in boundaries:
                for src in boundary.files:
                    if self._test_matches_source(test_path, src):
                        boundary.files.append(test_path)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                remaining_tests.append(test_path)

        # Distribute unmatched tests to the last boundary
        if remaining_tests and boundaries:
            boundaries[-1].files.extend(remaining_tests)

    def _apply_effort_multiplier(
        self,
        boundaries: list[SubtaskBoundary],
        workspace: str | Path,
    ) -> None:
        """Inflate estimated_lines for boundaries whose files already exist on disk.

        Modifying existing code is harder than creating new files: the engineer
        must read, understand context, edit carefully, and fix broken tests.
        """
        workspace_path = Path(workspace)
        for boundary in boundaries:
            existing_count = sum(
                1 for f in boundary.files
                if (workspace_path / f).exists()
            )
            if existing_count == 0:
                continue
            total = len(boundary.files) or 1
            existing_ratio = existing_count / total
            # Scale multiplier proportionally: all-existing gets full multiplier,
            # mixed gets a blend between 1.0 and MODIFICATION_EFFORT_MULTIPLIER.
            blended = 1.0 + (self.MODIFICATION_EFFORT_MULTIPLIER - 1.0) * existing_ratio
            boundary.estimated_lines = int(boundary.estimated_lines * blended)

    @staticmethod
    def _classify_boundary_tier(boundary: SubtaskBoundary) -> int:
        """Classify a boundary into an execution tier based on its file paths.

        Tier 0 = infrastructure (config, migrations, scripts, etc.)
        Tier 1 = source code (default)
        Tier 2 = tests
        """
        for file_path in boundary.files:
            for part in Path(file_path).parts:
                part_lower = part.lower()
                if part_lower in _TEST_DIRS:
                    return 2
                if part_lower in _INFRA_DIRS:
                    return 0
        return _DEFAULT_TIER

    @staticmethod
    def _scope_checklist_to_boundary(
        parent_checklist: list[dict], boundary_files: list[str]
    ) -> list[dict]:
        """Filter requirements checklist to items whose files overlap this boundary.

        Preserves original item IDs for cross-reference traceability.
        Items with no file associations are dropped — cross-cutting concerns
        are better evaluated at the fan-in level.
        """
        if not parent_checklist or not boundary_files:
            return []

        boundary_names = {Path(f).name.lower() for f in boundary_files}

        return [
            copy.deepcopy(item) for item in parent_checklist
            if any(
                Path(f).name.lower() in boundary_names
                for f in item.get("files", [])
            )
        ]

    def _infer_boundary_dependencies(
        self, boundaries: list[SubtaskBoundary]
    ) -> None:
        """Set depends_on_subtasks conservatively to maximize parallelism.

        Only adds a dependency when boundary j creates a file that boundary i
        imports (detected by filename reference in approach steps). Tier-based
        ordering is intentionally NOT used — co-location already pairs source
        with tests, so most boundaries can run in parallel.

        Preserves manually-set deps (e.g. from approach-step fallback).
        """
        if len(boundaries) < 2:
            return

        # Build a set of file stems per boundary for import-reference matching
        boundary_stems: list[set[str]] = [
            {Path(f).stem for f in b.files if not self._is_test_file(f)}
            for b in boundaries
        ]

        for i, boundary in enumerate(boundaries):
            # Skip boundaries that already have manually-set deps
            if boundary.depends_on_subtasks:
                continue

            deps: list[int] = []
            # Check if any approach step references a file owned by another boundary
            for j, other_stems in enumerate(boundary_stems):
                if i == j:
                    continue
                for step in boundary.approach_steps:
                    if any(stem in step for stem in other_stems if len(stem) > 3):
                        deps.append(j)
                        break

            boundary.depends_on_subtasks = deps

    def _extract_relevant_steps(
        self, approach_steps: list[str], files: list[str]
    ) -> list[str]:
        """
        Extract approach steps that are relevant to the given files.

        Simple heuristic: include steps that mention any of the file names.
        """
        relevant_steps = []
        file_names = {Path(f).name for f in files}

        for step in approach_steps:
            # Check if step mentions any of the file names
            if any(fname in step for fname in file_names):
                relevant_steps.append(step)

        return relevant_steps

    def _create_subtask(
        self, parent: Task, boundary: SubtaskBoundary, index: int, total: int,
        plan: PlanDocument | None = None,
    ) -> Task:
        """
        Build a child Task from a boundary.

        Args:
            parent: Parent task
            boundary: SubtaskBoundary defining this subtask's scope
            index: 1-based index (e.g., 1, 2, 3...)
            total: Total number of subtasks
            plan: The architect's plan (preferred over parent.plan which may be unset)

        Returns:
            A new Task configured as a subtask
        """
        subtask_id = f"{parent.id}-sub{index}"

        # Prefer the explicitly-passed plan, fall back to parent.plan
        source_plan = plan or parent.plan

        # Create scoped plan for this subtask
        subtask_plan = PlanDocument(
            objectives=(source_plan.objectives if source_plan and source_plan.objectives
                        else [f"Complete {boundary.name}"]) + [
                f"Scope: {boundary.name} ({len(boundary.files)} files)"
            ],
            approach=boundary.approach_steps,
            risks=source_plan.risks if source_plan else [],
            success_criteria=(source_plan.success_criteria if source_plan and source_plan.success_criteria
                              else [f"All changes in {len(boundary.files)} files are implemented and tested"]),
            files_to_modify=boundary.files,
            dependencies=source_plan.dependencies if source_plan else [],
        )

        # Build depends_on from boundary dependencies
        depends_on_task_ids = []
        for dep_index in boundary.depends_on_subtasks:
            dep_task_id = f"{parent.id}-sub{dep_index + 1}"
            depends_on_task_ids.append(dep_task_id)

        # Create subtask with inherited context
        subtask = Task(
            id=subtask_id,
            type=parent.type,
            status=TaskStatus.PENDING,
            priority=parent.priority,
            created_by=parent.created_by,
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title=f"{parent.title} - {boundary.name}",
            description=(
                f"Subtask {index}/{total} of {parent.id}\n\n"
                f"Goal: {parent.context.get('user_goal', parent.description)}\n\n"
                f"Scope: {boundary.name}\n\n"
                f"Files:\n"
                + "\n".join(f"- {f}" for f in boundary.files)
            ),
            depends_on=depends_on_task_ids,
            blocks=[],
            parent_task_id=parent.id,
            subtask_ids=[],
            decomposition_strategy=parent.decomposition_strategy,
            acceptance_criteria=[
                f"Complete implementation for {len(boundary.files)} files",
                "All tests pass",
            ],
            deliverables=boundary.files,
            notes=[
                f"Auto-generated subtask from parent {parent.id}",
                f"Estimated lines: {boundary.estimated_lines}",
            ],
            context={
                **copy.deepcopy(parent.context),
                "parent_task_id": parent.id,
                "subtask_index": index,
                "subtask_total": total,
            },
            plan=subtask_plan,
        )

        # Scope checklist to this subtask's file boundaries so self-eval
        # and prompt injection only reference in-scope deliverables
        parent_checklist = parent.context.get("requirements_checklist", [])
        if parent_checklist:
            scoped = self._scope_checklist_to_boundary(parent_checklist, boundary.files)
            if scoped:
                subtask.context["requirements_checklist"] = scoped
            else:
                subtask.context.pop("requirements_checklist", None)

        return subtask
