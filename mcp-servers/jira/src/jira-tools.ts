import JiraClient from "jira-client";

interface SearchIssuesArgs {
  jql: string;
  maxResults?: number;
  fields?: string[];
}

interface GetIssueArgs {
  issueKey: string;
}

interface CreateIssueArgs {
  project: string;
  summary: string;
  description: string;
  issueType: "Story" | "Bug" | "Task";
  labels?: string[];
}

interface CreateEpicArgs {
  project: string;
  title: string;
  description: string;
}

interface CreateSubtaskArgs {
  parentKey: string;
  summary: string;
  description: string;
}

interface TransitionIssueArgs {
  issueKey: string;
  transitionName: string;
}

interface AddCommentArgs {
  issueKey: string;
  comment: string;
}

interface UpdateFieldArgs {
  issueKey: string;
  fieldName: string;
  value: any;
}

interface CreateEpicWithSubtasksArgs {
  project: string;
  epicTitle: string;
  epicDescription: string;
  subtasks: Array<{
    summary: string;
    description: string;
  }>;
}

export async function searchIssues(client: JiraClient, args: SearchIssuesArgs) {
  const { jql, maxResults = 50, fields = ["key", "summary", "status"] } = args;

  const result = await client.searchJira(jql, {
    maxResults,
    fields,
  });

  return {
    success: true,
    data: {
      total: result.total,
      issues: result.issues.map((issue: any) => ({
        key: issue.key,
        summary: issue.fields.summary,
        status: issue.fields.status?.name,
        url: `https://${process.env.JIRA_SERVER}/browse/${issue.key}`,
      })),
    },
  };
}

export async function getIssue(client: JiraClient, args: GetIssueArgs) {
  const { issueKey } = args;

  const issue = await client.findIssue(issueKey);

  return {
    success: true,
    data: {
      key: issue.key,
      summary: issue.fields.summary,
      description: issue.fields.description,
      status: issue.fields.status?.name,
      issueType: issue.fields.issuetype?.name,
      assignee: issue.fields.assignee?.displayName,
      reporter: issue.fields.reporter?.displayName,
      created: issue.fields.created,
      updated: issue.fields.updated,
      url: `https://${process.env.JIRA_SERVER}/browse/${issue.key}`,
    },
  };
}

export async function createIssue(client: JiraClient, args: CreateIssueArgs) {
  const { project, summary, description, issueType, labels = [] } = args;

  const issueData = {
    fields: {
      project: { key: project },
      summary,
      description,
      issuetype: { name: issueType },
      labels,
    },
  };

  const result = await client.addNewIssue(issueData);

  return {
    success: true,
    data: {
      key: result.key,
      id: result.id,
      url: `https://${process.env.JIRA_SERVER}/browse/${result.key}`,
    },
  };
}

export async function createEpic(client: JiraClient, args: CreateEpicArgs) {
  const { project, title, description } = args;

  const epicData = {
    fields: {
      project: { key: project },
      summary: title,
      description,
      issuetype: { name: "Epic" },
    },
  };

  const result = await client.addNewIssue(epicData);

  return {
    success: true,
    data: {
      key: result.key,
      id: result.id,
      url: `https://${process.env.JIRA_SERVER}/browse/${result.key}`,
    },
  };
}

export async function createSubtask(client: JiraClient, args: CreateSubtaskArgs) {
  const { parentKey, summary, description } = args;

  const parent = await client.findIssue(parentKey);

  const subtaskData = {
    fields: {
      project: { key: parent.fields.project.key },
      summary,
      description,
      issuetype: { name: "Sub-task" },
      parent: { key: parentKey },
    },
  };

  const result = await client.addNewIssue(subtaskData);

  return {
    success: true,
    data: {
      key: result.key,
      id: result.id,
      parentKey,
      url: `https://${process.env.JIRA_SERVER}/browse/${result.key}`,
    },
  };
}

export async function transitionIssue(client: JiraClient, args: TransitionIssueArgs) {
  const { issueKey, transitionName } = args;

  const transitions = await client.listTransitions(issueKey);
  const transition = transitions.transitions.find(
    (t: any) => t.name.toLowerCase() === transitionName.toLowerCase()
  );

  if (!transition) {
    throw new Error(
      `Transition '${transitionName}' not found. Available: ${transitions.transitions
        .map((t: any) => t.name)
        .join(", ")}`
    );
  }

  await client.transitionIssue(issueKey, {
    transition: { id: transition.id },
  });

  const issue = await client.findIssue(issueKey);

  return {
    success: true,
    data: {
      issueKey,
      newStatus: issue.fields.status?.name,
    },
  };
}

export async function addComment(client: JiraClient, args: AddCommentArgs) {
  const { issueKey, comment } = args;

  await client.addComment(issueKey, comment);

  return {
    success: true,
    data: {
      issueKey,
      comment,
    },
  };
}

export async function updateField(client: JiraClient, args: UpdateFieldArgs) {
  const { issueKey, fieldName, value } = args;

  await client.updateIssue(issueKey, {
    fields: {
      [fieldName]: value,
    },
  });

  return {
    success: true,
    data: {
      issueKey,
      fieldName,
      value,
    },
  };
}

export async function createEpicWithSubtasks(
  client: JiraClient,
  args: CreateEpicWithSubtasksArgs
) {
  const { project, epicTitle, epicDescription, subtasks } = args;

  const epicResult = await createEpic(client, {
    project,
    title: epicTitle,
    description: epicDescription,
  });

  const epicKey = epicResult.data.key;
  const createdSubtasks = [];

  for (const subtask of subtasks) {
    const subtaskResult = await createSubtask(client, {
      parentKey: epicKey,
      summary: subtask.summary,
      description: subtask.description,
    });
    createdSubtasks.push(subtaskResult.data);
  }

  return {
    success: true,
    data: {
      epic: epicResult.data,
      subtasks: createdSubtasks,
    },
  };
}
