declare module 'jira-client' {
  interface JiraClientOptions {
    protocol: string;
    host: string;
    username: string;
    password: string;
    apiVersion: string;
    strictSSL: boolean;
  }

  interface JiraIssue {
    id: string;
    key: string;
    fields: {
      summary: string;
      description: string;
      status: { name: string } | null;
      issuetype: { name: string } | null;
      assignee: { displayName: string } | null;
      reporter: { displayName: string } | null;
      project: { key: string };
      created: string;
      updated: string;
      [key: string]: any;
    };
  }

  interface SearchResult {
    total: number;
    issues: JiraIssue[];
  }

  interface Transition {
    id: string;
    name: string;
  }

  interface TransitionsResult {
    transitions: Transition[];
  }

  interface CreateIssueResult {
    id: string;
    key: string;
  }

  class JiraClient {
    constructor(options: JiraClientOptions);

    searchJira(jql: string, options?: { maxResults?: number; fields?: string[] }): Promise<SearchResult>;
    findIssue(issueKey: string): Promise<JiraIssue>;
    addNewIssue(issue: any): Promise<CreateIssueResult>;
    updateIssue(issueKey: string, update: any): Promise<void>;
    listTransitions(issueKey: string): Promise<TransitionsResult>;
    transitionIssue(issueKey: string, transition: any): Promise<void>;
    addComment(issueKey: string, comment: string): Promise<void>;
  }

  export = JiraClient;
}
