const USER_MEMORY_DIRECTORIES = [
  { directory: 'profile', type: 'profile', singleton: true },
  { directory: 'preferences', type: 'preference' },
  { directory: 'entities', type: 'entity' },
  { directory: 'events', type: 'event' },
  { directory: 'cases', type: 'case' },
  { directory: 'patterns', type: 'pattern' },
  { directory: 'session_summaries', type: 'session_summary' },
  { directory: 'session_archives', type: 'session_archive' },
];
const AGENT_MEMORY_DIRECTORIES = [
  { directory: 'patterns', type: 'pattern' },
  { directory: 'tools', type: 'tool_memory' },
];
const TYPE_LABELS = {
  profile: '用户画像', preference: '偏好', entity: '实体', event: '事件', case: '案例',
  pattern: '模式', tool_memory: '工具记忆', session_summary: '会话摘要',
  session_archive: '会话归档', skill: 'Skill',
};

const state = {
  apiKey: sessionStorage.getItem('ogmemory-viewer-api-key') || '',
  accountId: localStorage.getItem('ogmemory-viewer-account-id') || '',
  tree: null, expanded: new Set(), directories: new Map(), directorySpecs: new Map(),
  singletonSpecs: new Map(), selectedUri: '', detailRequestId: 0, searchQuery: '', view: 'memory',
  adminMode: true,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  account: $('#account-select'), connection: $('#connection'), tree: $('#hierarchy-tree'),
  treeCount: $('#tree-count'), treeRoot: $('#tree-root'), search: $('#search-input'),
  refresh: $('#refresh-button'), settings: $('#settings-dialog'), detailView: $('#detail-view'),
  detailPlaceholder: $('#detail-placeholder'), shell: $('.app-shell'), pageTitle: $('#page-title'),
  memoryView: $('#memory-view'), dashboardView: $('#dashboard-view'),
};

function requestHeaders() { return state.apiKey ? { 'X-API-Key': state.apiKey } : {}; }
function apiUrl(path, values = {}) {
  const url = new URL(path, location.origin);
  Object.entries(values).forEach(([key, value]) => { if (value) url.searchParams.set(key, value); });
  return url;
}
async function fetchJson(url) {
  const response = await fetch(url, { headers: requestHeaders() });
  if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
  return response.json();
}

async function loadTree() {
  setLoading(true);
  try {
    const scope = await fetchJson(apiUrl('/api/v1/memory-viewer/scope'));
    state.adminMode = Boolean(scope.admin_mode);
    const body = state.adminMode
      ? await fetchJson(apiUrl('/api/v1/memory-viewer/admin/tree', { accountId: state.accountId }))
      : publicTree(scope);
    state.tree = body;
    state.accountId = body.account_id || '';
    state.directories.clear();
    clearDetail();
    localStorage.setItem('ogmemory-viewer-account-id', state.accountId);
    renderAccounts(body.accounts || []);
    openRootDirectories();
    renderContextTree();
    renderDashboard();
    setConnection(true);
  } catch (error) {
    showTreeError(error.message);
    setConnection(false);
    if (!state.apiKey) openSettings();
  } finally {
    setLoading(false);
  }
}

function publicTree(scope) {
  const accountId = String(scope.account_id || '');
  const userId = String(scope.user_id || '');
  const agents = (scope.agents || []).map((agent) => ({
    agent_id: String(agent.agent_id || ''), owner_user_id: userId,
  }));
  return {
    accounts: accountId ? [{ account_id: accountId }] : [],
    account_id: accountId,
    users: userId ? [{ user_id: userId, role: 'user', agents }] : [],
    agents,
  };
}

function renderAccounts(accounts) {
  elements.account.innerHTML = accounts.map((account) => {
    const id = String(account.account_id || '');
    const suffix = account.user_count === undefined ? '' : ` · ${account.user_count} 用户`;
    return `<option value="${escapeHtml(id)}">${escapeHtml(id + suffix)}</option>`;
  }).join('');
  elements.account.value = state.accountId;
  elements.account.disabled = accounts.length <= 1;
}
function openRootDirectories() {
  const root = contextRoot();
  state.expanded = new Set([root, `${root}/users`, `${root}/agents`]);
}
function contextRoot() { return state.accountId ? `ctx://${state.accountId}` : ''; }

function renderContextTree() {
  const root = contextRoot();
  const users = state.tree?.users || [];
  const agents = state.tree?.agents || users.flatMap((user) => (
    user.agents || []
  ).map((agent) => ({ ...agent, owner_user_id: user.user_id })));
  state.directorySpecs.clear();
  state.singletonSpecs.clear();
  elements.treeRoot.textContent = root || '无可用账户';
  elements.treeCount.textContent = `${users.length} users · ${agents.length} agents`;
  if (!root) {
    elements.tree.innerHTML = '<div class="tree-empty">无可见 ContextFS 节点</div>';
    return;
  }
  const usersRoot = `${root}/users`;
  const agentsRoot = `${root}/agents`;
  const usersHtml = users.map((user) => renderUserNode(user, usersRoot)).join('');
  const agentsHtml = agents.map((agent) => renderAgentNode(agent, agentsRoot)).join('');
  elements.tree.innerHTML = folderNode(root, state.accountId, 0, 'account', [
    folderNode(usersRoot, 'users', 1, 'folder', usersHtml, `${users.length}`),
    folderNode(agentsRoot, 'agents', 1, 'folder', agentsHtml, `${agents.length}`),
  ].join(''));
}

function renderUserNode(user, usersRoot) {
  const userId = String(user.user_id);
  const userRoot = `${usersRoot}/${encodeURIComponent(userId)}`;
  const memoriesRoot = `${userRoot}/memories`;
  const categories = USER_MEMORY_DIRECTORIES.map((entry) => {
    const key = `${memoriesRoot}/${entry.directory}`;
    const spec = { accountId: state.accountId, userId, agentId: '', type: entry.type };
    return entry.singleton
      ? singletonNode(key, entry.directory, 4, spec)
      : directoryNode(key, entry.directory, 4, spec);
  }).join('');
  const memories = folderNode(memoriesRoot, 'memories', 3, 'folder', categories);
  return folderNode(userRoot, userId, 2, 'user', memories, user.role || 'user');
}

function renderAgentNode(agent, agentsRoot) {
  const agentId = String(agent.agent_id);
  const ownerUserId = String(agent.owner_user_id || '');
  const agentRoot = `${agentsRoot}/${encodeURIComponent(agentId)}`;
  const memoriesRoot = `${agentRoot}/memories`;
  const memories = AGENT_MEMORY_DIRECTORIES.map((entry) => directoryNode(
    `${memoriesRoot}/${entry.directory}`, entry.directory, 4,
    { accountId: state.accountId, userId: ownerUserId, agentId, type: entry.type },
  )).join('');
  const skillsRoot = `${agentRoot}/skills`;
  const skills = ['generated', 'imported'].map((source) => directoryNode(
    `${skillsRoot}/${source}`, source, 4,
    { accountId: state.accountId, userId: ownerUserId, agentId, type: 'skill', source },
  )).join('');
  const children = [
    folderNode(memoriesRoot, 'memories', 3, 'folder', memories),
    folderNode(skillsRoot, 'skills', 3, 'folder', skills),
  ].join('');
  return folderNode(agentRoot, agentId, 2, 'agent', children, ownerUserId || 'owner unknown');
}

function folderNode(key, label, depth, kind, children, meta = '') {
  const expanded = state.expanded.has(key);
  return `<div class="context-node context-folder ${expanded ? 'open' : ''}" data-node-key="${escapeHtml(key)}">
    <button class="context-row" type="button" data-folder-key="${escapeHtml(key)}" style="--tree-depth:${depth}">
      <span class="context-chevron">${expanded ? '⌄' : '›'}</span><span class="context-icon" data-kind="${kind}">${kind === 'account' ? '▱' : '▰'}</span>
      <span class="context-name">${escapeHtml(label)}</span>${meta ? `<span class="context-meta">${escapeHtml(meta)}</span>` : ''}
    </button><div class="context-children">${children}</div></div>`;
}
function directoryNode(key, label, depth, spec) {
  state.directorySpecs.set(key, spec);
  const record = state.directories.get(key);
  const count = record ? `${record.items.length}${record.hasMore ? '+' : ''}` : '';
  return folderNode(key, label, depth, 'folder', renderDirectoryChildren(key, spec, record), count);
}
function singletonNode(key, label, depth, spec) {
  state.singletonSpecs.set(key, spec);
  const record = state.directories.get(key);
  const selected = record?.items?.[0]?.uri === state.selectedUri;
  return `<button class="context-row context-resource ${selected ? 'active' : ''}" type="button" data-singleton-key="${escapeHtml(key)}" style="--tree-depth:${depth}">
    <span class="context-chevron"></span><span class="context-icon" data-kind="profile">◇</span>
    <span class="context-name">${escapeHtml(label)}</span>${record?.loading ? '<span class="context-meta">…</span>' : ''}
  </button>`;
}
function renderDirectoryChildren(key, spec, record) {
  if (!state.expanded.has(key)) return '';
  if (!record) return '<div class="tree-empty nested">展开后加载</div>';
  if (record.loading && !record.items.length) return '<div class="tree-empty nested">加载中…</div>';
  if (record.error) return `<div class="tree-error">${escapeHtml(record.error)}</div>`;
  const items = record.items.filter(matchesSearch).map((memory) => renderMemoryNode(memory, spec)).join('');
  const empty = items ? '' : '<div class="tree-empty nested">没有节点</div>';
  const more = record.hasMore
    ? `<button class="tree-more" type="button" data-load-more="${escapeHtml(key)}">加载更多</button>`
    : '';
  return items + empty + more;
}
function renderMemoryNode(memory, spec) {
  const label = memory.type === 'profile' ? 'profile' : memory.id || 'unnamed';
  const selected = memory.uri === state.selectedUri;
  return `<button class="context-row context-resource ${selected ? 'active' : ''}" type="button"
    data-resource-uri="${escapeHtml(memory.uri)}" data-user="${escapeHtml(spec.userId)}" data-agent="${escapeHtml(spec.agentId)}" style="--tree-depth:5">
    <span class="context-chevron"></span><span class="context-icon" data-kind="${escapeHtml(memory.type)}">◇</span>
    <span class="context-name">${escapeHtml(label)}</span><span class="context-meta">${escapeHtml(typeLabel(memory.type))}</span>
  </button>`;
}
function matchesSearch(memory) {
  if (!state.searchQuery) return true;
  return [memory.id, memory.abstract, memory.context_uri, memory.uri, memory.type]
    .join(' ').toLocaleLowerCase().includes(state.searchQuery);
}

async function toggleDirectory(key) {
  if (state.expanded.has(key)) {
    state.expanded.delete(key);
    renderContextTree();
    return;
  }
  state.expanded.add(key);
  renderContextTree();
  const spec = state.directorySpecs.get(key);
  if (spec && !state.directories.has(key)) await loadDirectory(key, spec, null, false);
}
async function loadDirectory(key, spec, cursor, append) {
  if (!spec.userId) {
    const record = { items: [], hasMore: false, nextCursor: null, loading: false, error: 'Agent 缺少 owner_user_id' };
    state.directories.set(key, record);
    renderContextTree();
    return record;
  }
  const previous = state.directories.get(key);
  const requestId = (previous?.requestId || 0) + 1;
  state.directories.set(key, {
    items: previous?.items || [], hasMore: previous?.hasMore || false,
    nextCursor: previous?.nextCursor || null, loading: true, error: '', requestId,
  });
  renderContextTree();
  try {
    const path = state.adminMode ? '/api/v1/memory-viewer/admin/memories' : '/api/v1/memories';
    const body = await fetchJson(apiUrl(path, {
      accountId: state.adminMode ? spec.accountId : '', userId: state.adminMode ? spec.userId : '', agentId: spec.agentId,
      type: spec.type, source: spec.source || '', cursor: cursor || '', limit: '100', sort: 'name',
    }));
    if (state.directories.get(key)?.requestId !== requestId) return state.directories.get(key);
    const items = append ? [...(previous?.items || []), ...body.memories] : body.memories;
    const record = {
      items, hasMore: body.truncated, nextCursor: body.next_cursor,
      loading: false, error: '', requestId,
    };
    state.directories.set(key, record);
    renderContextTree();
    renderDashboard();
    return record;
  } catch (error) {
    if (state.directories.get(key)?.requestId === requestId) {
      state.directories.set(key, {
        items: previous?.items || [], hasMore: previous?.hasMore || false,
        nextCursor: previous?.nextCursor || null, loading: false, error: error.message, requestId,
      });
      renderContextTree();
    }
    return state.directories.get(key);
  }
}
async function openSingleton(key) {
  const spec = state.singletonSpecs.get(key);
  if (!spec) return;
  let record = state.directories.get(key);
  if (!record) record = await loadDirectory(key, spec, null, false);
  const memory = record?.items?.[0];
  if (memory) await openDetail(memory.uri, spec.userId, spec.agentId);
}

async function openDetail(uri, userId, agentId) {
  const requestId = ++state.detailRequestId;
  state.selectedUri = uri;
  renderContextTree();
  elements.detailPlaceholder.hidden = true;
  elements.detailView.hidden = false;
  $('#detail-type').textContent = '加载中';
  $('#detail-type').dataset.type = '';
  $('#detail-title').textContent = '读取节点';
  $('#detail-uri').textContent = uri;
  $('#detail-abstract').textContent = '读取中';
  $('#detail-overview').textContent = '读取中';
  $('#detail-content').textContent = '读取中';
  $('#detail-metadata').innerHTML = '<div class="metadata-empty">读取中</div>';
  try {
    const path = state.adminMode ? '/api/v1/memory-viewer/admin/detail' : '/api/v1/memories/detail';
    const memory = await fetchJson(apiUrl(path, {
      accountId: state.adminMode ? state.accountId : '', userId: state.adminMode ? userId : '', agentId, uri,
    }));
    if (requestId !== state.detailRequestId || uri !== state.selectedUri) return;
    $('#detail-type').textContent = typeLabel(memory.type);
    $('#detail-type').dataset.type = memory.type;
    $('#detail-title').textContent = memory.id || '未命名节点';
    $('#detail-uri').textContent = memory.context_uri || memory.uri;
    $('#detail-abstract').textContent = memory.abstract || '暂无摘要';
    $('#detail-overview').textContent = displayValue(memory.overview, '暂无概览');
    $('#detail-content').textContent = displayValue(memory.content, '暂无正文');
    renderMetadata(memory.metadata || {});
  } catch (error) {
    if (requestId === state.detailRequestId && uri === state.selectedUri) {
      $('#detail-title').textContent = '读取失败';
      $('#detail-content').textContent = error.message;
    }
  }
}
function clearDetail() {
  state.detailRequestId += 1;
  state.selectedUri = '';
  elements.detailView.hidden = true;
  elements.detailPlaceholder.hidden = false;
}
function renderMetadata(metadata) {
  const entries = Object.entries(metadata);
  $('#detail-metadata').innerHTML = entries.length ? entries.map(([key, value]) => `
    <div class="metadata-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(displayValue(value, '—'))}</strong></div>`).join('') : '<div class="metadata-empty">暂无元数据</div>';
}

function renderDashboard() {
  const users = state.tree?.users || [];
  const agents = state.tree?.agents || users.flatMap((user) => user.agents || []);
  const loaded = [...state.directories.values()].reduce((total, record) => total + record.items.length, 0);
  const skills = [...state.directories.values()].reduce((total, record) => (
    total + record.items.filter((item) => item.type === 'skill').length
  ), 0);
  $('#dashboard-users').textContent = users.length;
  $('#dashboard-agents').textContent = agents.length;
  $('#dashboard-memories').textContent = loaded;
  $('#dashboard-skills').textContent = skills;
}
function switchView(view) {
  state.view = view;
  document.querySelectorAll('[data-view]').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === view);
    if (item.matches('.sidebar-link')) item.toggleAttribute('aria-current', item.dataset.view === view);
  });
  const dashboard = view === 'dashboard';
  elements.dashboardView.hidden = !dashboard;
  elements.memoryView.hidden = dashboard;
  elements.pageTitle.textContent = dashboard ? '控制台' : view === 'skill' ? 'Skill' : 'ContextFS';
  if (dashboard) { renderDashboard(); return; }
  if (view === 'skill') {
    const root = contextRoot();
    state.expanded.add(root);
    state.expanded.add(`${root}/agents`);
    (state.tree?.agents || []).forEach((agent) => {
      const agentRoot = `${root}/agents/${encodeURIComponent(agent.agent_id)}`;
      state.expanded.add(agentRoot);
      state.expanded.add(`${agentRoot}/skills`);
    });
  }
  renderContextTree();
}

function showTreeError(message) {
  elements.treeRoot.textContent = '加载失败';
  elements.treeCount.textContent = '';
  elements.tree.innerHTML = `<div class="tree-error root-error">${escapeHtml(message)}</div>`;
}
function setLoading(loading) { elements.refresh.classList.toggle('loading', loading); }
function setConnection(online) {
  elements.connection.className = `connection sidebar-connection ${online ? 'online' : 'offline'}`;
  elements.connection.innerHTML = `<span></span>${online ? '已连接' : '连接失败'}`;
}
function displayValue(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}
function typeLabel(type) { return TYPE_LABELS[type] || type; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

elements.account.addEventListener('change', () => { state.accountId = elements.account.value; loadTree(); });
document.querySelector('.app-sidebar').addEventListener('click', (event) => {
  const view = event.target.closest('[data-view]');
  if (view) switchView(view.dataset.view);
});
elements.dashboardView.addEventListener('click', (event) => {
  const view = event.target.closest('[data-view]');
  if (view) switchView(view.dataset.view);
});
elements.tree.addEventListener('click', async (event) => {
  const more = event.target.closest('[data-load-more]');
  if (more) {
    const key = more.dataset.loadMore;
    const record = state.directories.get(key);
    const spec = state.directorySpecs.get(key);
    if (record?.nextCursor && spec) await loadDirectory(key, spec, record.nextCursor, true);
    return;
  }
  const resource = event.target.closest('[data-resource-uri]');
  if (resource) {
    await openDetail(resource.dataset.resourceUri, resource.dataset.user, resource.dataset.agent);
    return;
  }
  const singleton = event.target.closest('[data-singleton-key]');
  if (singleton) {
    await openSingleton(singleton.dataset.singletonKey);
    return;
  }
  const folder = event.target.closest('[data-folder-key]');
  if (folder) await toggleDirectory(folder.dataset.folderKey);
});
elements.search.addEventListener('input', () => {
  state.searchQuery = elements.search.value.trim().toLocaleLowerCase();
  renderContextTree();
});
elements.refresh.addEventListener('click', loadTree);
$('#theme-button').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('ogmemory-viewer-theme', next);
});
$('#sidebar-toggle').addEventListener('click', () => {
  elements.shell.classList.add('sidebar-collapsed');
  localStorage.setItem('ogmemory-viewer-sidebar', 'collapsed');
});
$('.workspace-icon').addEventListener('click', () => {
  if (!elements.shell.classList.contains('sidebar-collapsed')) return;
  elements.shell.classList.remove('sidebar-collapsed');
  localStorage.setItem('ogmemory-viewer-sidebar', 'expanded');
});
function openSettings() { $('#api-key-input').value = state.apiKey; elements.settings.showModal(); }
$('#settings-button').addEventListener('click', openSettings);
$('#save-settings').addEventListener('click', (event) => {
  event.preventDefault();
  state.apiKey = $('#api-key-input').value.trim();
  sessionStorage.setItem('ogmemory-viewer-api-key', state.apiKey);
  elements.settings.close();
  loadTree();
});
elements.settings.addEventListener('click', (event) => {
  if (event.target === elements.settings) elements.settings.close();
});

document.documentElement.dataset.theme = localStorage.getItem('ogmemory-viewer-theme') || 'light';
elements.shell.classList.toggle('sidebar-collapsed', localStorage.getItem('ogmemory-viewer-sidebar') === 'collapsed');
loadTree();
