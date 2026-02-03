import { Octokit } from "@octokit/rest";

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
