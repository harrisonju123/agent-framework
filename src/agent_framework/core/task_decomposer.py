"""Task decomposition logic for splitting large plans into subtasks."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional
from pathlib import Path

from agent_framework.core.task import Task, PlanDocument, TaskType, TaskStatus


def estimate_plan_lines(plan: "PlanDocument") -> int:
    """Estimate total implementation lines from plan signals.

    Combines file count (50 lines/file) and approach step count (25 lines/step).
    Used by both task decomposition and budget sizing.
    """
    file_estimate = len(plan.files_to_modify) * 50 if plan.files_to_modify else 0
    step_estimate = len(plan.approach) * 25 if plan.approach else 0
    return file_estimate + step_estimate


@dataclass
class SubtaskBoundary:
    """A natural split point identified in a plan."""

    name: str  # e.g. "Database schema changes"
    files: list[str]  # Files this subtask covers
    approach_steps: list[str]  # Relevant steps from parent plan
    depends_on_subtasks: list[int]  # Indices of subtasks this depends on (for ordering)
    estimated_lines: int


class TaskDecomposer:
    """Decomposes large tasks into smaller, independent subtasks."""

    # Configuration constants
    DECOMPOSE_THRESHOLD = 500  # Lines above which decomposition triggers
    TARGET_SUBTASK_SIZE = 250  # Target lines per subtask
    MIN_SUBTASK_SIZE = 50  # Don't create subtasks smaller than this
    MAX_SUBTASKS = 5  # Cap on number of subtasks
    MAX_DEPTH = 1  # Subtasks cannot themselves decompose

    def should_decompose(self, plan: PlanDocument, estimated_lines: int) -> bool:
        """
        Determine if a task should be decomposed into subtasks.

        Args:
            plan: The task's plan document
            estimated_lines: Estimated total lines of code for the task

        Returns:
            True if task exceeds threshold and has enough files to split
        """
        # Check if we're above the threshold
        if estimated_lines < self.DECOMPOSE_THRESHOLD:
            return False

        # Need at least 2 files to meaningfully split
        if len(plan.files_to_modify) < 2:
            return False

        return True

    def decompose(
        self, parent_task: Task, plan: PlanDocument, estimated_lines: int
    ) -> list[Task]:
        """
        Split parent task into independent subtasks.

        Args:
            parent_task: The task to decompose
            plan: The task's plan document
            estimated_lines: Estimated total lines of code

        Returns:
            List of created subtask objects

        Side effects:
            Updates parent_task.subtask_ids with the IDs of created subtasks
        """
        # Check if parent is already a subtask (max depth = 1)
        if parent_task.parent_task_id is not None:
            return []

        # Identify natural split boundaries
        boundaries = self._identify_split_boundaries(plan, estimated_lines)

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
            )
            subtasks.append(subtask)

        # Update parent with subtask IDs
        parent_task.subtask_ids = [task.id for task in subtasks]

        return subtasks

    def _identify_split_boundaries(
        self, plan: PlanDocument, estimated_lines: int
    ) -> list[SubtaskBoundary]:
        """
        Group files_to_modify into clusters that form natural subtasks.

        Uses directory-prefix grouping as the primary heuristic:
        - Group files by their top-level directory (e.g., all src/core/ together)
        - If a group is still >300 lines, split further by subdirectories
        - If we end up with < 2 groups, fall back to splitting approach steps

        Args:
            plan: The task's plan document
            estimated_lines: Total estimated lines for the task

        Returns:
            List of SubtaskBoundary objects representing natural splits
        """
        files = plan.files_to_modify
        if not files:
            return []

        # Estimate lines per file (rough heuristic: total / file count)
        lines_per_file = estimated_lines // len(files) if files else 0

        # Group files by top-level directory
        dir_groups: dict[str, list[str]] = {}
        for file in files:
            # Extract top-level directory (e.g., "src/core/task.py" -> "src")
            parts = Path(file).parts
            top_dir = parts[0] if parts else "root"
            if top_dir not in dir_groups:
                dir_groups[top_dir] = []
            dir_groups[top_dir].append(file)

        # Create boundaries from directory groups
        boundaries = []
        for dir_name, group_files in dir_groups.items():
            group_estimated = len(group_files) * lines_per_file

            # If group is still too large, try to split by subdirectory
            if group_estimated > 300 and len(group_files) > 2:
                sub_boundaries = self._split_by_subdirectory(
                    group_files, lines_per_file, dir_name
                )
                boundaries.extend(sub_boundaries)
            else:
                # Create single boundary for this directory group
                boundary = SubtaskBoundary(
                    name=f"Changes in {dir_name}",
                    files=group_files,
                    approach_steps=self._extract_relevant_steps(
                        plan.approach, group_files
                    ),
                    depends_on_subtasks=[],  # Assume independent for now
                    estimated_lines=group_estimated,
                )
                boundaries.append(boundary)

        # If we still have < 2 boundaries, fall back to approach step splitting
        if len(boundaries) < 2:
            boundaries = self._split_by_approach_steps(plan, estimated_lines)

        # Filter out boundaries that are too small
        boundaries = [b for b in boundaries if b.estimated_lines >= self.MIN_SUBTASK_SIZE]

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

        # If no specific matches, include all steps (conservative approach)
        if not relevant_steps:
            relevant_steps = approach_steps

        return relevant_steps

    def _create_subtask(
        self, parent: Task, boundary: SubtaskBoundary, index: int, total: int
    ) -> Task:
        """
        Build a child Task from a boundary.

        Args:
            parent: Parent task
            boundary: SubtaskBoundary defining this subtask's scope
            index: 1-based index (e.g., 1, 2, 3...)
            total: Total number of subtasks

        Returns:
            A new Task configured as a subtask
        """
        subtask_id = f"{parent.id}-sub{index}"

        # Create scoped plan for this subtask
        subtask_plan = PlanDocument(
            objectives=[f"Complete {boundary.name} for parent task"],
            approach=boundary.approach_steps,
            risks=parent.plan.risks if parent.plan else [],
            success_criteria=[
                f"All changes in {len(boundary.files)} files are implemented and tested"
            ],
            files_to_modify=boundary.files,
            dependencies=parent.plan.dependencies if parent.plan else [],
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
            assigned_to=parent.assigned_to,
            created_at=datetime.now(UTC),
            title=f"{parent.title} - {boundary.name}",
            description=f"Subtask {index}/{total} of {parent.id}\n\n{boundary.name}\n\nFiles:\n"
            + "\n".join(f"- {f}" for f in boundary.files),
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
                **parent.context,
                "parent_task_id": parent.id,
                "subtask_index": index,
                "subtask_total": total,
            },
            plan=subtask_plan,
        )

        return subtask
