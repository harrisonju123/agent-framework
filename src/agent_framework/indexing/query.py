"""Query interface for injecting index data into prompts (M3)."""


class IndexQuery:

    def query_for_prompt(
        self, repo_slug: str, task_goal: str, max_chars: int = 4000
    ) -> str:
        raise NotImplementedError
