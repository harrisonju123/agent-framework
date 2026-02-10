import { Octokit } from "@octokit/rest";
import { execSync } from "child_process";

interface CreateBranchArgs {
  owner: string;
  repo: string;
  branchName: string;
  fromBranch?: string;
}

interface CreatePRArgs {
  owner: string;
  repo: string;
  title: string;
  body: string;
  head: string;
  base?: string;
  draft?: boolean;
  labels?: string[];
}

interface AddPRCommentArgs {
  owner: string;
  repo: string;
  prNumber: number;
  body: string;
}

interface GetPRByBranchArgs {
  owner: string;
  repo: string;
  branchName: string;
}

interface LinkPRToJiraArgs {
  owner: string;
  repo: string;
  prNumber: number;
  jiraKey: string;
}

interface CloneRepoArgs {
  owner: string;
  repo: string;
  localPath: string;
}

interface CommitChangesArgs {
  localPath: string;
  message: string;
}

interface PushBranchArgs {
  localPath: string;
  branchName: string;
}

interface GetPRArgs {
  owner: string;
  repo: string;
  prNumber: number;
}

interface GetPRCommentsArgs {
  owner: string;
  repo: string;
  prNumber: number;
}

interface GetPRDiffArgs {
  owner: string;
  repo: string;
  prNumber: number;
}

interface GetCheckRunsArgs {
  owner: string;
  repo: string;
  ref: string;
}

interface UpdatePRArgs {
  owner: string;
  repo: string;
  prNumber: number;
  title?: string;
  body?: string;
  state?: "open" | "closed";
}

export async function createBranch(octokit: Octokit, args: CreateBranchArgs) {
  const { owner, repo, branchName, fromBranch = "main" } = args;

  const { data: ref } = await octokit.git.getRef({
    owner,
    repo,
    ref: `heads/${fromBranch}`,
  });

  const { data: newRef } = await octokit.git.createRef({
    owner,
    repo,
    ref: `refs/heads/${branchName}`,
    sha: ref.object.sha,
  });

  return {
    success: true,
    data: {
      name: branchName,
      sha: newRef.object.sha,
      url: `https://github.com/${owner}/${repo}/tree/${branchName}`,
    },
  };
}

export async function createPR(octokit: Octokit, args: CreatePRArgs) {
  const { owner, repo, title, body, head, base = "main", draft = false, labels = [] } = args;

  const { data: pr } = await octokit.pulls.create({
    owner,
    repo,
    title,
    body,
    head,
    base,
    draft,
  });

  if (labels.length > 0) {
    await octokit.issues.addLabels({
      owner,
      repo,
      issue_number: pr.number,
      labels,
    });
  }

  return {
    success: true,
    data: {
      number: pr.number,
      url: pr.html_url,
      html_url: pr.html_url,
    },
  };
}

export async function addPRComment(octokit: Octokit, args: AddPRCommentArgs) {
  const { owner, repo, prNumber, body } = args;

  await octokit.issues.createComment({
    owner,
    repo,
    issue_number: prNumber,
    body,
  });

  return {
    success: true,
    data: {
      prNumber,
      body,
    },
  };
}

export async function getPRByBranch(octokit: Octokit, args: GetPRByBranchArgs) {
  const { owner, repo, branchName } = args;

  const { data: pulls } = await octokit.pulls.list({
    owner,
    repo,
    head: `${owner}:${branchName}`,
    state: "all",
  });

  if (pulls.length === 0) {
    return {
      success: true,
      data: null,
    };
  }

  const pr = pulls[0];

  return {
    success: true,
    data: {
      number: pr.number,
      title: pr.title,
      state: pr.state,
      url: pr.html_url,
    },
  };
}

export async function linkPRToJira(octokit: Octokit, args: LinkPRToJiraArgs) {
  const { owner, repo, prNumber, jiraKey } = args;

  const { data: pr } = await octokit.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
  });

  const jiraServer = process.env.JIRA_SERVER || "jira.example.com";
  const jiraLink = `https://${jiraServer}/browse/${jiraKey}`;

  const updatedBody = pr.body
    ? `${pr.body}\n\n---\nJIRA: [${jiraKey}](${jiraLink})`
    : `JIRA: [${jiraKey}](${jiraLink})`;

  await octokit.pulls.update({
    owner,
    repo,
    pull_number: prNumber,
    body: updatedBody,
  });

  return {
    success: true,
    data: {
      prNumber,
      jiraKey,
      jiraLink,
    },
  };
}

export async function cloneRepo(args: CloneRepoArgs) {
  const { owner, repo, localPath } = args;
  const token = process.env.GITHUB_TOKEN;
  const cloneUrl = `https://x-access-token:${token}@github.com/${owner}/${repo}.git`;

  execSync(`git clone ${cloneUrl} ${localPath}`, { timeout: 300000 });
  return { success: true, data: { path: localPath, owner, repo } };
}

export async function commitChanges(args: CommitChangesArgs) {
  const { localPath, message } = args;
  execSync("git add -A", { cwd: localPath, timeout: 30000 });

  // Check if there are changes to commit
  const status = execSync("git status --porcelain", {
    cwd: localPath,
    encoding: "utf-8",
  });
  if (!status.trim()) {
    return { success: true, data: { message: "No changes to commit" } };
  }

  execSync(`git commit -m "${message.replace(/"/g, '\\"')}"`, {
    cwd: localPath,
    timeout: 30000,
  });
  return { success: true, data: { message } };
}

export async function pushBranch(args: PushBranchArgs) {
  const { localPath, branchName } = args;
  execSync(`git push -u origin ${branchName}`, {
    cwd: localPath,
    timeout: 120000,
  });
  return { success: true, data: { branch: branchName } };
}

export async function getPR(octokit: Octokit, args: GetPRArgs) {
  const { owner, repo, prNumber } = args;

  const { data: pr } = await octokit.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
  });

  const { data: files } = await octokit.pulls.listFiles({
    owner,
    repo,
    pull_number: prNumber,
    per_page: 100,
  });

  return {
    success: true,
    data: {
      number: pr.number,
      title: pr.title,
      state: pr.state,
      url: pr.html_url,
      head: pr.head.ref,
      base: pr.base.ref,
      body: pr.body,
      additions: pr.additions,
      deletions: pr.deletions,
      changed_files: pr.changed_files,
      mergeable: pr.mergeable,
      files_truncated: pr.changed_files > files.length,
      files: files.map((f) => ({
        filename: f.filename,
        status: f.status,
        additions: f.additions,
        deletions: f.deletions,
        patch: f.patch,
      })),
    },
  };
}

export async function getPRComments(octokit: Octokit, args: GetPRCommentsArgs) {
  const { owner, repo, prNumber } = args;

  const { data: issueComments } = await octokit.issues.listComments({
    owner,
    repo,
    issue_number: prNumber,
    per_page: 100,
  });

  const { data: reviewComments } = await octokit.pulls.listReviewComments({
    owner,
    repo,
    pull_number: prNumber,
    per_page: 100,
  });

  return {
    success: true,
    data: {
      issue_comments_truncated: issueComments.length >= 100,
      issue_comments: issueComments.map((c) => ({
        id: c.id,
        user: c.user?.login,
        body: c.body,
        created_at: c.created_at,
      })),
      review_comments_truncated: reviewComments.length >= 100,
      review_comments: reviewComments.map((c) => ({
        id: c.id,
        user: c.user?.login,
        body: c.body,
        path: c.path,
        line: c.line,
        created_at: c.created_at,
      })),
    },
  };
}

export async function getPRDiff(octokit: Octokit, args: GetPRDiffArgs) {
  const { owner, repo, prNumber } = args;

  // mediaType "diff" makes Octokit return a raw string, but the
  // type system still sees the pulls.get response shape — double cast needed.
  const { data: diff } = await octokit.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
    mediaType: { format: "diff" },
  });

  return {
    success: true,
    data: {
      diff: diff as unknown as string,
    },
  };
}

export async function getCheckRuns(octokit: Octokit, args: GetCheckRunsArgs) {
  const { owner, repo, ref } = args;

  const { data } = await octokit.checks.listForRef({
    owner,
    repo,
    ref,
    per_page: 100,
  });

  return {
    success: true,
    data: {
      total_count: data.total_count,
      check_runs: data.check_runs.map((c) => ({
        id: c.id,
        name: c.name,
        status: c.status,
        conclusion: c.conclusion,
        started_at: c.started_at,
        completed_at: c.completed_at,
        html_url: c.html_url,
        output: c.output
          ? {
              title: c.output.title,
              summary: c.output.summary && c.output.summary.length > 500
                ? c.output.summary.substring(0, 500) + "...[truncated]"
                : c.output.summary,
            }
          : null,
      })),
    },
  };
}

export async function updatePR(octokit: Octokit, args: UpdatePRArgs) {
  const { owner, repo, prNumber, title, body, state } = args;

  const updateFields: {
    title?: string;
    body?: string;
    state?: "open" | "closed";
  } = {};
  if (title !== undefined) updateFields.title = title;
  if (body !== undefined) updateFields.body = body;
  if (state !== undefined) updateFields.state = state;

  const { data: pr } = await octokit.pulls.update({
    owner,
    repo,
    pull_number: prNumber,
    ...updateFields,
  });

  return {
    success: true,
    data: {
      number: pr.number,
      title: pr.title,
      state: pr.state,
      url: pr.html_url,
    },
  };
}
