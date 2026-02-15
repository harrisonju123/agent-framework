"""Task decomposition logic for splitting large tasks into independent subtasks."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional

from agent_framework.core.task import PlanDocument, Task, TaskStatus, TaskType


@dataclass
class SubtaskBoundary:
    """A natural split point identified in a plan."""

    name: str  # e.g. "Database schema changes"
    files: list[str]  # Files this subtask covers
    approach_steps: list[str]  # Relevant steps from parent plan
    depends_on_subtasks: list[int]  # Indices of subtasks this depends on (for ordering)
    estimated_lines: int


class TaskDecomposer:
    """Splits large tasks into independent subtasks based on natural boundaries."""

    # Configuration
    DECOMPOSE_THRESHOLD = 500  # Lines above which decomposition triggers
    TARGET_SUBTASK_SIZE = 250  # Target lines per subtask
    MIN_SUBTASK_SIZE = 50  # Don't create subtasks smaller than this
    MAX_SUBTASKS = 5  # Cap on number of subtasks
    MAX_DEPTH = 1  # Subtasks cannot themselves decompose
    LARGE_BOUNDARY_THRESHOLD = 300  # Boundary size requiring further splitting

    def should_decompose(self, plan: PlanDocument, estimated_lines: int) -> bool:
        """Determine if task should be decomposed based on size and structure.

        Returns True if:
        - Task exceeds DECOMPOSE_THRESHOLD lines
        - Plan has multiple files to modify (allowing natural splits)
        """
        if estimated_lines < self.DECOMPOSE_THRESHOLD:
            return False

        # Need at least 2 files to have a meaningful split
        if len(plan.files_to_modify) < 2:
            return False

        return True

    def decompose(self, parent_task: Task, plan: PlanDocument, estimated_lines: int) -> list[Task]:
        """Split parent into independent subtasks.

        Returns the created subtask list and updates parent_task.subtask_ids.
        Respects MAX_DEPTH to prevent nested decomposition.
        """
        # Prevent nested decomposition
        if parent_task.parent_task_id is not None:
            return []

        # Identify natural split boundaries
        boundaries = self._identify_split_boundaries(plan)

        # Cap at MAX_SUBTASKS
        if len(boundaries) > self.MAX_SUBTASKS:
            boundaries = boundaries[:self.MAX_SUBTASKS]

        # Create subtasks
        subtasks = []
        for idx, boundary in enumerate(boundaries):
            subtask = self._create_subtask(
                parent=parent_task,
                boundary=boundary,
                index=idx,
                total=len(boundaries)
            )
            subtasks.append(subtask)

        # Update parent with subtask IDs
        parent_task.subtask_ids = [task.id for task in subtasks]

        return subtasks

    def _identify_split_boundaries(self, plan: PlanDocument) -> list[SubtaskBoundary]:
        """Group files_to_modify into clusters that form natural subtasks.

        Strategy:
        1. Group files by top-level directory
        2. If a group >300 lines estimated, split further
        3. If total groups < 2, split approach steps into halves
        """
        # Group files by top-level directory
        dir_groups: dict[str, list[str]] = {}
        for file_path in plan.files_to_modify:
            # Extract top-level directory (e.g., "src/core/" from "src/core/task.py")
            parts = file_path.split('/')
            if len(parts) > 1:
                top_dir = f"{parts[0]}/{parts[1]}/" if len(parts) > 2 else f"{parts[0]}/"
            else:
                top_dir = "root/"

            if top_dir not in dir_groups:
                dir_groups[top_dir] = []
            dir_groups[top_dir].append(file_path)

        # If only one group, try to split by file count
        if len(dir_groups) == 1:
            single_group = list(dir_groups.values())[0]
            if len(single_group) >= 4:
                # Split into two groups
                mid = len(single_group) // 2
                dir_groups = {
                    "part_1/": single_group[:mid],
                    "part_2/": single_group[mid:]
                }

        # Create boundaries from groups
        boundaries = []
        for dir_name, files in dir_groups.items():
            # Estimate lines per file (simple heuristic: 100 lines per file)
            estimated = len(files) * 100

            # If group is too large, split further
            if estimated > self.LARGE_BOUNDARY_THRESHOLD and len(files) > 1:
                mid = len(files) // 2
                boundaries.append(SubtaskBoundary(
                    name=f"{dir_name} (part 1)",
                    files=files[:mid],
                    approach_steps=self._partition_approach_steps(plan.approach, len(boundaries), len(dir_groups) * 2),
                    depends_on_subtasks=[],
                    estimated_lines=len(files[:mid]) * 100
                ))
                boundaries.append(SubtaskBoundary(
                    name=f"{dir_name} (part 2)",
                    files=files[mid:],
                    approach_steps=self._partition_approach_steps(plan.approach, len(boundaries), len(dir_groups) * 2),
                    depends_on_subtasks=[],
                    estimated_lines=len(files[mid:]) * 100
                ))
            else:
                boundaries.append(SubtaskBoundary(
                    name=dir_name.rstrip('/'),
                    files=files,
                    approach_steps=self._partition_approach_steps(plan.approach, len(boundaries), len(dir_groups)),
                    depends_on_subtasks=[],
                    estimated_lines=estimated
                ))

        # Filter out boundaries that are too small
        boundaries = [b for b in boundaries if b.estimated_lines >= self.MIN_SUBTASK_SIZE]

        # If still < 2 boundaries, split approach steps
        if len(boundaries) < 2 and len(plan.approach) >= 4:
            mid = len(plan.approach) // 2
            boundaries = [
                SubtaskBoundary(
                    name="Implementation phase 1",
                    files=plan.files_to_modify[:len(plan.files_to_modify)//2] if plan.files_to_modify else [],
                    approach_steps=plan.approach[:mid],
                    depends_on_subtasks=[],
                    estimated_lines=len(plan.files_to_modify) * 50
                ),
                SubtaskBoundary(
                    name="Implementation phase 2",
                    files=plan.files_to_modify[len(plan.files_to_modify)//2:] if plan.files_to_modify else [],
                    approach_steps=plan.approach[mid:],
                    depends_on_subtasks=[],
                    estimated_lines=len(plan.files_to_modify) * 50
                )
            ]

        return boundaries

    def _partition_approach_steps(self, approach: list[str], current_idx: int, total_partitions: int) -> list[str]:
        """Partition approach steps proportionally to boundaries."""
        if not approach or total_partitions == 0:
            return []

        steps_per_partition = max(1, len(approach) // total_partitions)
        start = current_idx * steps_per_partition
        end = start + steps_per_partition if current_idx < total_partitions - 1 else len(approach)

        return approach[start:end]

    def _create_subtask(
        self,
        parent: Task,
        boundary: SubtaskBoundary,
        index: int,
        total: int
    ) -> Task:
        """Build a child Task from a boundary.

        - ID pattern: {parent.id}-sub-{index}
        - Sets parent_task_id to parent.id
        - Inherits parent's context
        - Creates a PlanDocument scoped to boundary's files and approach_steps
        - Sets depends_on based on boundary.depends_on_subtasks
        """
        subtask_id = f"{parent.id}-sub-{index}"

        # Create scoped plan for subtask
        subtask_plan = PlanDocument(
            objectives=[f"Subtask {index + 1}/{total}: {boundary.name}"],
            approach=boundary.approach_steps,
            risks=parent.plan.risks if parent.plan else [],
            success_criteria=[f"Complete changes for: {', '.join(boundary.files[:3])}{'...' if len(boundary.files) > 3 else ''}"],
            files_to_modify=boundary.files,
            dependencies=parent.plan.dependencies if parent.plan else []
        )

        # Build depends_on from boundary
        depends_on = [f"{parent.id}-sub-{dep_idx}" for dep_idx in boundary.depends_on_subtasks]

        # Create subtask
        subtask = Task(
            id=subtask_id,
            type=parent.type,
            status=TaskStatus.PENDING,
            priority=parent.priority,
            created_by=parent.created_by,
            assigned_to=parent.assigned_to,
            created_at=datetime.now(UTC),
            title=f"{parent.title} - {boundary.name}",
            description=f"Subtask {index + 1}/{total} of parent task {parent.id}\n\n{boundary.name}",
            depends_on=depends_on,
            context={
                **parent.context,
                "parent_task_id": parent.id,
                "subtask_index": index,
                "subtask_total": total,
            },
            parent_task_id=parent.id,
            plan=subtask_plan
        )

        return subtask
