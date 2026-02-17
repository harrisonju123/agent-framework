import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'dashboard', component: () => import('../pages/DashboardPage.vue'), meta: { title: 'Dashboard' } },
  { path: '/agents', name: 'agents', component: () => import('../pages/AgentsPage.vue'), meta: { title: 'Agents' } },
  { path: '/tasks', name: 'tasks', component: () => import('../pages/TasksPage.vue'), meta: { title: 'Tasks' } },
  { path: '/logs', name: 'logs', component: () => import('../pages/LogsPage.vue'), meta: { title: 'Logs' } },
  { path: '/settings', name: 'settings', component: () => import('../pages/SettingsPage.vue'), meta: { title: 'Settings' } },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

export default createRouter({ history: createWebHistory(), routes })
