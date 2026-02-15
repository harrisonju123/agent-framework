"""Engineer specialization system based on file patterns.

Selects specialized engineer profiles (backend, frontend, infrastructure) based on
file patterns detected in tasks.
"""

import fnmatch
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .task import Task


# Extensions that indicate source code (not config, docs, etc.)
KNOWN_SOURCE_EXTENSIONS = {
    "go", "py", "rb", "java", "rs", "c", "cpp", "cs",
    "tsx", "jsx", "vue", "svelte",
    "ts", "js",
    "css", "scss", "sass", "less",
    "tf", "tfvars",
    "sql", "sh",
}


@dataclass
class SpecializationProfile:
    """Profile defining an engineer specialization."""

    id: str
    name: str
    description: str
    file_patterns: List[str]  # Glob patterns to match
    prompt_suffix: str  # Additional context appended to base engineer prompt
    teammates: Dict[str, Dict[str, str]]  # Specialized teammates
    tool_guidance: str  # Tool-specific guidance for this specialization


# Define specialization profiles
BACKEND_PROFILE = SpecializationProfile(
    id="backend",
    name="Backend Engineer",
    description="Specializes in server-side logic, APIs, databases, and backend services",
    file_patterns=[
        "**/*.go",
        "**/*.py",
        "**/*.rb",
        "**/*.java",
        "**/*.rs",
        "**/cmd/**",
        "**/internal/**",
        "**/pkg/**",
        "**/api/**",
        "**/handlers/**",
        "**/controllers/**",
        "**/models/**",
        "**/services/**",
        "**/repositories/**",
        "**/*_test.go",
        "**/*_test.py",
        "**/tests/**",
        "**/migrations/**",
        "**/*.sql",
    ],
    prompt_suffix="""
BACKEND SPECIALIZATION:
You are a Backend Engineer specializing in server-side development.

FOCUS AREAS:
- API design and implementation (REST, GraphQL, gRPC)
- Database schema design and queries
- Business logic and service layer
- Authentication and authorization
- Performance optimization (caching, query optimization)
- Concurrency and parallel processing
- Error handling and logging
- Integration with external services

BEST PRACTICES:
- Follow language-specific idioms (Go: effective Go, Python: PEP8, etc.)
- Use parameterized queries to prevent SQL injection
- Implement proper error handling and logging
- Write comprehensive unit and integration tests
- Consider performance implications (N+1 queries, memory usage)
- Document API contracts with examples
""",
    teammates={
        "database-expert": {
            "description": "Expert in database design, query optimization, and migrations",
            "prompt": "You are a database expert. Review schema designs, query performance, indexing strategies, and migration scripts. Identify N+1 queries, missing indexes, and inefficient joins. Suggest optimizations for database operations."
        },
        "api-reviewer": {
            "description": "Reviews API design for consistency, RESTful principles, and error handling",
            "prompt": "You are an API design expert. Review endpoint design, HTTP status codes, request/response schemas, error handling, and API documentation. Ensure consistency, proper versioning, and adherence to REST/GraphQL best practices."
        }
    },
    tool_guidance="""
COMMON TOOLS FOR BACKEND:
- Testing: go test, pytest, rspec, jest (for Node.js)
- Linting: golangci-lint, pylint, rubocop
- DB tools: migrate, sqlc, psql, mysql
- API testing: curl, httpie, postman
- Profiling: pprof (Go), py-spy (Python)
"""
)

FRONTEND_PROFILE = SpecializationProfile(
    id="frontend",
    name="Frontend Engineer",
    description="Specializes in user interfaces, client-side logic, and web applications",
    file_patterns=[
        # Framework-specific extensions are strong frontend signals
        "**/*.tsx",
        "**/*.jsx",
        "**/*.vue",
        "**/*.svelte",
        # Style files
        "**/*.css",
        "**/*.scss",
        "**/*.sass",
        "**/*.less",
        # Directory-based signals (strong indicators)
        "**/components/**",
        "**/pages/**",
        "**/views/**",
        "**/styles/**",
        "**/public/**",
        "**/src/app/**",
        "**/src/components/**",
        # Test files with frontend-specific extensions
        "**/*.test.tsx",
        "**/*.test.jsx",
        "**/*.spec.tsx",
        "**/*.spec.jsx",
    ],
    prompt_suffix="""
FRONTEND SPECIALIZATION:
You are a Frontend Engineer specializing in user interface development.

FOCUS AREAS:
- Component design and composition
- State management (Redux, Context, Zustand, etc.)
- User experience and accessibility (a11y)
- Performance optimization (lazy loading, code splitting, memoization)
- Responsive design and cross-browser compatibility
- Form handling and validation
- API integration and data fetching
- Client-side routing

BEST PRACTICES:
- Follow framework conventions (React, Vue, Angular, etc.)
- Implement proper error boundaries and loading states
- Ensure accessibility (ARIA labels, keyboard navigation, screen reader support)
- Optimize bundle size and load times
- Write component tests (React Testing Library, Vue Test Utils, etc.)
- Avoid prop drilling with proper state management
- Use semantic HTML and proper CSS architecture
- Prevent XSS vulnerabilities (sanitize user input, use framework safeguards)
""",
    teammates={
        "ux-reviewer": {
            "description": "Reviews user experience, accessibility, and interaction patterns",
            "prompt": "You are a UX expert. Review component design for usability, accessibility (WCAG compliance), interaction patterns, and user feedback. Check for proper ARIA labels, keyboard navigation, focus management, and loading/error states."
        },
        "performance-auditor": {
            "description": "Reviews frontend performance and bundle optimization",
            "prompt": "You are a frontend performance expert. Review code for performance issues: unnecessary re-renders, large bundle sizes, missing code splitting, unoptimized images, blocking resources. Suggest optimizations like lazy loading, memoization, and caching strategies."
        }
    },
    tool_guidance="""
COMMON TOOLS FOR FRONTEND:
- Testing: jest, vitest, cypress, playwright, react-testing-library
- Linting: eslint, prettier, stylelint
- Build tools: webpack, vite, rollup, esbuild
- Package managers: npm, yarn, pnpm
- Dev tools: browser DevTools, React DevTools, Vue DevTools
- Accessibility: axe, lighthouse, WAVE
"""
)

INFRASTRUCTURE_PROFILE = SpecializationProfile(
    id="infrastructure",
    name="Infrastructure Engineer",
    description="Specializes in DevOps, CI/CD, infrastructure as code, and deployment",
    file_patterns=[
        # Container files
        "**/Dockerfile*",
        "Dockerfile*",
        "**/docker-compose*.yml",
        "docker-compose*.yml",
        # Terraform
        "**/*.tf",
        "**/*.tfvars",
        "**/terraform/**",
        # CI/CD pipelines
        ".github/workflows/**",
        "**/.github/workflows/**",
        ".gitlab-ci.yml",
        "**/.gitlab-ci.yml",
        "**/Jenkinsfile",
        # Kubernetes / Helm — use both anchored and unanchored forms
        # because Path.match matches from the right
        "k8s/**",
        "kubernetes/**",
        "helm/**",
        "charts/**",
        "deployment/**",
        "manifests/**",
        # Config management
        "ansible/**",
        "playbooks/**",
        # Build / scripts
        "**/Makefile",
        "**/nginx.conf",
        "**/*.sh",
        # Monitoring
        "prometheus/**",
        "grafana/**",
        "monitoring/**",
    ],
    prompt_suffix="""
INFRASTRUCTURE SPECIALIZATION:
You are an Infrastructure Engineer specializing in DevOps and deployment.

FOCUS AREAS:
- Container orchestration (Docker, Kubernetes)
- CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- Infrastructure as Code (Terraform, CloudFormation, Pulumi)
- Configuration management (Ansible, Chef, Puppet)
- Monitoring and observability (Prometheus, Grafana, ELK)
- Cloud platforms (AWS, GCP, Azure)
- Networking and security
- Scaling and high availability

BEST PRACTICES:
- Follow infrastructure as code best practices
- Implement proper secret management (never commit secrets)
- Use multi-stage Docker builds to optimize image size
- Implement health checks and readiness probes
- Set resource limits and requests for containers
- Use declarative configuration over imperative scripts
- Implement proper logging and monitoring
- Follow least-privilege security principles
- Version infrastructure code and test changes
- Document deployment procedures and runbooks
""",
    teammates={
        "security-hardening": {
            "description": "Reviews infrastructure security, secrets management, and hardening",
            "prompt": "You are an infrastructure security expert. Review configurations for security issues: exposed secrets, excessive permissions, missing security policies, vulnerable dependencies, insecure defaults. Suggest security hardening measures."
        },
        "sre-reviewer": {
            "description": "Reviews reliability, observability, and operational excellence",
            "prompt": "You are an SRE expert. Review infrastructure for reliability: proper monitoring, alerting, logging, health checks, backup strategies, disaster recovery, and scaling considerations. Ensure operational excellence."
        }
    },
    tool_guidance="""
COMMON TOOLS FOR INFRASTRUCTURE:
- Containers: docker, docker-compose, podman
- Orchestration: kubectl, helm, kustomize
- IaC: terraform, terragrunt, pulumi
- CI/CD: github-actions, gitlab-ci, jenkins
- Cloud CLIs: aws-cli, gcloud, az
- Config mgmt: ansible, ansible-playbook
- Monitoring: prometheus, grafana, datadog
- Scripts: bash, make
"""
)


# All available profiles
SPECIALIZATION_PROFILES = [
    BACKEND_PROFILE,
    FRONTEND_PROFILE,
    INFRASTRUCTURE_PROFILE,
]


def detect_file_patterns(task: Task) -> List[str]:
    """Extract file paths from task context and plan.

    Returns:
        List of file paths mentioned in the task
    """
    files = []

    # Check plan.files_to_modify
    if task.plan and task.plan.files_to_modify:
        files.extend(task.plan.files_to_modify)

    # Check context for file references
    context = task.context or {}

    # Check for explicit file lists
    if "files" in context:
        files.extend(context["files"])

    if "files_to_modify" in context:
        files.extend(context["files_to_modify"])

    # Check structured_findings for file references
    if "structured_findings" in context:
        findings = context["structured_findings"]
        if isinstance(findings, dict) and "findings" in findings:
            for finding in findings["findings"]:
                if "file" in finding and finding["file"]:
                    files.append(finding["file"])

    # Parse file paths from description text
    # Match paths like src/path/to/file.ext — require known source extension
    # to avoid false positives on version strings, URLs, etc.
    description_text = f"{task.title} {task.description}"
    ext_pattern = "|".join(re.escape(e) for e in sorted(KNOWN_SOURCE_EXTENSIONS))
    file_pattern = rf'\b[\w\-\.]+/[\w\-\./]+\.({ext_pattern})\b'
    matches = re.findall(file_pattern, description_text)
    # re.findall with groups returns just the group — reconstruct full matches
    for match in re.finditer(file_pattern, description_text):
        files.append(match.group(0))

    return list(set(files))  # Deduplicate


def _matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if a file path matches a glob pattern.

    Uses fnmatch for matching, with a fallback for **/ prefix patterns
    so root-level files (e.g. 'handler.go') can still match '**/*.go'.
    """
    if fnmatch.fnmatch(file_path, pattern):
        return True
    # **/ prefix requires at least one directory — strip it for root-level files
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(file_path, pattern[3:])
    return False


def match_patterns(files: List[str], patterns: List[str]) -> int:
    """Count how many files match the given glob patterns.

    Args:
        files: List of file paths to check
        patterns: List of glob patterns to match against

    Returns:
        Number of matching files
    """
    matches = 0

    for file_path in files:
        for pattern in patterns:
            if _matches_pattern(file_path, pattern):
                matches += 1
                break  # Count each file only once

    return matches


def detect_specialization(task: Task) -> Optional[SpecializationProfile]:
    """Detect the appropriate engineer specialization based on task file patterns.

    Analyzes files mentioned in the task and returns the specialization profile
    with the highest number of matching patterns.

    Args:
        task: Task to analyze

    Returns:
        SpecializationProfile if a clear match is found, None for generic engineer
    """
    # Extract files from task
    files = detect_file_patterns(task)

    if not files:
        return None

    # Score each specialization profile
    scores = []
    for profile in SPECIALIZATION_PROFILES:
        match_count = match_patterns(files, profile.file_patterns)
        if match_count > 0:
            scores.append((match_count, profile))

    if not scores:
        return None

    # Sort by match count (descending)
    scores.sort(reverse=True, key=lambda x: x[0])

    top_score, top_profile = scores[0]
    total_files = len(files)

    # Require clear signal: at least 2 matching files AND >50% of files
    if top_score >= max(2, total_files * 0.5):
        return top_profile

    return None


def apply_specialization_to_prompt(
    base_prompt: str,
    profile: Optional[SpecializationProfile]
) -> str:
    """Apply specialization context to the base engineer prompt.

    Args:
        base_prompt: Base engineer prompt from agents.yaml
        profile: Specialization profile to apply (None for generic)

    Returns:
        Enhanced prompt with specialization context
    """
    if not profile:
        return base_prompt

    return f"{base_prompt}\n\n{profile.prompt_suffix}\n\n{profile.tool_guidance}"


def get_specialized_teammates(
    base_teammates: Dict[str, Any],
    profile: Optional[SpecializationProfile]
) -> Dict[str, Any]:
    """Merge base teammates with specialization-specific teammates.

    Args:
        base_teammates: Base teammates from agent definition
        profile: Specialization profile (None for generic)

    Returns:
        Merged teammates dictionary
    """
    if not profile:
        return base_teammates

    merged = dict(base_teammates)
    merged.update(profile.teammates)

    return merged
