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
  singletonSpecs: new Map(), selectedUri: '', selectedDirectoryKey: '', detailRequestId: 0, searchQuery: '', view: 'memory',
  adminMode: true, retrievalRequestId: 0, searchTargets: new Map(), lastSearchTarget: null,
  retrievalHits: [],
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  account: $('#account-select'), connection: $('#connection'), tree: $('#hierarchy-tree'),
  treeCount: $('#tree-count'), treeRoot: $('#tree-root'), search: $('#search-input'),
  refresh: $('#refresh-button'), settings: $('#settings-dialog'), detailView: $('#detail-view'),
  detailPlaceholder: $('#detail-placeholder'), shell: $('.app-shell'), pageTitle: $('#page-title'),
  memoryView: $('#memory-view'), dashboardView: $('#dashboard-view'),
  retrievalView: $('#retrieval-view'), retrievalScope: $('#retrieval-scope'),
  retrievalResults: $('#retrieval-results'), retrievalDetail: $('#retrieval-detail'),
  revealCurrent: $('#reveal-current-button'),
};

function requestHeaders() { return state.apiKey ? { 'X-API-Key': state.apiKey } : {}; }
function apiUrl(path, values = {}) {
  const url = new URL(path, location.origin);
  Object.entries(values).forEach(([key, value]) => { if (value) url.searchParams.set(key, value); });
  return url;
}
async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { ...requestHeaders(), ...(options.headers || {}) },
  });
  if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
  return response.json();
}

async function loadTree() {
  resetRetrieval();
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
    renderSearchScopes();
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
function renderSearchScopes() {
  const previous = elements.retrievalScope.value;
  state.searchTargets.clear();
  state.searchTargets.set('all', { accountId: state.accountId, userId: '', agentId: '' });
  const options = ['<option value="all">全部可见记忆</option>'];
  (state.tree?.users || []).forEach((user, userIndex) => {
    const userId = String(user.user_id || '');
    const userKey = `user-${userIndex}`;
    state.searchTargets.set(userKey, { accountId: state.accountId, userId, agentId: '' });
    options.push(`<option value="${userKey}">${escapeHtml(userId)} · 用户记忆</option>`);
    (user.agents || []).forEach((agent, agentIndex) => {
      const agentId = String(agent.agent_id || '');
      const agentKey = `agent-${userIndex}-${agentIndex}`;
      state.searchTargets.set(agentKey, { accountId: state.accountId, userId, agentId });
      options.push(`<option value="${agentKey}">${escapeHtml(agentId)} · Agent</option>`);
    });
  });
  elements.retrievalScope.innerHTML = options.join('') || '<option value="">无可检索范围</option>';
  elements.retrievalScope.value = state.searchTargets.has(previous) ? previous : 'all';
}
function openRootDirectories() {
  const root = contextRoot();
  state.expanded = new Set([root, `${root}/users`, `${root}/agents`]);
  if (state.adminMode) state.expanded.add('ctx://');
}
function contextRoot() { return state.accountId ? `ctx://${state.accountId}` : ''; }

function renderContextTree() {
  const accountRoot = contextRoot();
  const users = state.tree?.users || [];
  const agents = state.tree?.agents || users.flatMap((user) => (
    user.agents || []
  ).map((agent) => ({ ...agent, owner_user_id: user.user_id })));
  state.directorySpecs.clear();
  state.singletonSpecs.clear();
  elements.treeRoot.textContent = state.adminMode ? 'ctx://' : accountRoot || '无可用账户';
  elements.treeCount.textContent = state.adminMode
    ? `${state.tree?.accounts?.length || 0} accounts`
    : `${users.length} users · ${agents.length} agents`;
  elements.revealCurrent.disabled = !state.selectedUri || !state.selectedDirectoryKey;
  elements.revealCurrent.title = state.selectedDirectoryKey || '定位当前文件';
  if (!accountRoot) {
    elements.tree.innerHTML = '<div class="tree-empty">无可见 ContextFS 节点</div>';
    return;
  }
  if (state.view === 'skill') {
    renderSkillTree(accountRoot, agents);
    return;
  }
  const branchDepth = state.adminMode ? 2 : 1;
  const usersRoot = `${accountRoot}/users`;
  const agentsRoot = `${accountRoot}/agents`;
  const usersHtml = users.map((user) => renderUserNode(user, usersRoot, branchDepth + 1)).join('');
  const agentsHtml = agents.map((agent) => renderAgentNode(agent, agentsRoot, branchDepth + 1)).join('');
  const accountChildren = [
    folderNode(usersRoot, 'users', branchDepth, 'folder', usersHtml, `${users.length}`),
    folderNode(agentsRoot, 'agents', branchDepth, 'folder', agentsHtml, `${agents.length}`),
  ].join('');
  elements.tree.innerHTML = renderTreeRoot(accountRoot, accountChildren, users.length, agents.length);
}

function renderTreeRoot(accountRoot, accountChildren, userCount, agentCount) {
  if (!state.adminMode) return folderNode(accountRoot, state.accountId, 0, 'account', accountChildren);
  const accountRows = (state.tree?.accounts || []).map((account) => {
    const accountId = String(account.account_id || '');
    const meta = accountId === state.accountId
      ? `${userCount} users · ${agentCount} agents`
      : account.user_count === undefined ? '' : `${account.user_count} users`;
    return accountId === state.accountId
      ? folderNode(accountRoot, accountId, 1, 'account', accountChildren, meta)
      : accountSwitchNode(accountId, meta);
  }).join('');
  return folderNode('ctx://', 'ctx://', 0, 'root', accountRows, `${state.tree?.accounts?.length || 0}`);
}

function accountSwitchNode(accountId, meta) {
  return `<button class="context-row" type="button" data-account-switch="${escapeHtml(accountId)}" style="--tree-depth:1">
    <span class="context-chevron">›</span><span class="context-icon" data-kind="account">▱</span>
    <span class="context-name">${escapeHtml(accountId)}</span>${meta ? `<span class="context-meta">${escapeHtml(meta)}</span>` : ''}
  </button>`;
}

function renderSkillTree(accountRoot, agents) {
  const branchDepth = state.adminMode ? 2 : 1;
  const agentsRoot = `${accountRoot}/agents`;
  const agentsHtml = agents.map((agent) => renderSkillAgentNode(agent, agentsRoot, branchDepth + 1)).join('');
  const accountChildren = folderNode(agentsRoot, 'agents', branchDepth, 'folder', agentsHtml, `${agents.length}`);
  elements.tree.innerHTML = renderTreeRoot(accountRoot, accountChildren, state.tree?.users?.length || 0, agents.length);
}

function renderSkillAgentNode(agent, agentsRoot, depth = 2) {
  const agentId = String(agent.agent_id);
  const ownerUserId = String(agent.owner_user_id || '');
  const agentRoot = `${agentsRoot}/${encodeURIComponent(agentId)}`;
  const skillsRoot = `${agentRoot}/skills`;
  const skills = ['generated', 'imported'].map((source) => directoryNode(
    `${skillsRoot}/${source}`, source, depth + 2,
    { accountId: state.accountId, userId: ownerUserId, agentId, type: 'skill', source, itemDepth: depth + 3 },
  )).join('');
  return folderNode(agentRoot, agentId, depth, 'agent',
    folderNode(skillsRoot, 'skills', depth + 1, 'folder', skills), ownerUserId || 'owner unknown');
}

function renderUserNode(user, usersRoot, depth = 2) {
  const userId = String(user.user_id);
  const userRoot = `${usersRoot}/${encodeURIComponent(userId)}`;
  const memoriesRoot = `${userRoot}/memories`;
  const categories = USER_MEMORY_DIRECTORIES.map((entry) => {
    const key = `${memoriesRoot}/${entry.directory}`;
    const spec = { accountId: state.accountId, userId, agentId: '', type: entry.type, itemDepth: depth + 3 };
    return entry.singleton
      ? singletonNode(key, entry.directory, depth + 2, spec)
      : directoryNode(key, entry.directory, depth + 2, spec);
  }).join('');
  const memories = folderNode(memoriesRoot, 'memories', depth + 1, 'folder', categories);
  return folderNode(userRoot, userId, depth, 'user', memories, user.role || 'user');
}

function renderAgentNode(agent, agentsRoot, depth = 2) {
  const agentId = String(agent.agent_id);
  const ownerUserId = String(agent.owner_user_id || '');
  const agentRoot = `${agentsRoot}/${encodeURIComponent(agentId)}`;
  const memoriesRoot = `${agentRoot}/memories`;
  const memories = AGENT_MEMORY_DIRECTORIES.map((entry) => directoryNode(
    `${memoriesRoot}/${entry.directory}`, entry.directory, depth + 2,
    { accountId: state.accountId, userId: ownerUserId, agentId, type: entry.type, itemDepth: depth + 3 },
  )).join('');
  const skillsRoot = `${agentRoot}/skills`;
  const skills = ['generated', 'imported'].map((source) => directoryNode(
    `${skillsRoot}/${source}`, source, depth + 2,
    { accountId: state.accountId, userId: ownerUserId, agentId, type: 'skill', source, itemDepth: depth + 3 },
  )).join('');
  const children = [
    folderNode(memoriesRoot, 'memories', depth + 1, 'folder', memories),
    folderNode(skillsRoot, 'skills', depth + 1, 'folder', skills),
  ].join('');
  return folderNode(agentRoot, agentId, depth, 'agent', children, ownerUserId || 'owner unknown');
}

function folderNode(key, label, depth, kind, children, meta = '') {
  const expanded = state.expanded.has(key);
  return `<div class="context-node context-folder ${expanded ? 'open' : ''}" data-node-key="${escapeHtml(key)}">
    <button class="context-row" type="button" data-folder-key="${escapeHtml(key)}" style="--tree-depth:${depth}">
      <span class="context-chevron">${expanded ? '⌄' : '›'}</span><span class="context-icon" data-kind="${kind}">${kind === 'root' ? '⌂' : kind === 'account' ? '▱' : '▰'}</span>
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
    data-resource-uri="${escapeHtml(memory.uri)}" data-user="${escapeHtml(spec.userId)}" data-agent="${escapeHtml(spec.agentId)}" style="--tree-depth:${spec.itemDepth || 5}">
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
  state.selectedDirectoryKey = key;
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
  state.selectedDirectoryKey = '';
  elements.detailView.hidden = true;
  elements.detailPlaceholder.hidden = false;
}
function revealSelectedResource() {
  if (!state.selectedDirectoryKey || !state.selectedUri) return;
  const accountRoot = contextRoot();
  if (state.adminMode) state.expanded.add('ctx://');
  state.expanded.add(accountRoot);
  let current = accountRoot;
  state.selectedDirectoryKey.slice(accountRoot.length).split('/').filter(Boolean).forEach((segment) => {
    current = `${current}/${segment}`;
    state.expanded.add(current);
  });
  renderContextTree();
  requestAnimationFrame(() => {
    const selected = elements.tree.querySelector('.context-resource.active');
    if (!selected) return;
    selected.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' });
    selected.focus({ preventScroll: true });
  });
}
function renderMetadata(metadata) {
  renderMetadataInto($('#detail-metadata'), metadata);
}
function renderMetadataInto(container, metadata) {
  const entries = Object.entries(metadata).filter(([, value]) => value !== undefined && value !== null && value !== '');
  container.innerHTML = entries.length ? entries.map(([key, value]) => `
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
async function runRetrieval() {
  const query = $('#retrieval-input').value.trim();
  const target = state.searchTargets.get(elements.retrievalScope.value);
  if (!query || !target) return;
  const requestId = ++state.retrievalRequestId;
  state.lastSearchTarget = target;
  clearRetrievalDetail();
  $('#retrieval-submit').disabled = true;
  $('#retrieval-result-count').textContent = '';
  elements.retrievalResults.innerHTML = '<div class="retrieval-empty loading"><span>⌕</span><strong>正在检索</strong><p>匹配语义相关记忆…</p></div>';
  const type = $('#retrieval-type').value;
  const threshold = $('#retrieval-threshold').value;
  const payload = {
    query,
    top_k: Number($('#retrieval-count').value),
    accountId: target.accountId,
    userId: target.userId,
    agentId: target.agentId,
    types: type ? [type] : undefined,
    score_threshold: threshold === '' ? undefined : Number(threshold),
    sessionId: $('#retrieval-session').value.trim() || undefined,
  };
  try {
    const body = await fetchJson('/api/v1/memory-viewer/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (requestId !== state.retrievalRequestId) return;
    renderRetrievalResults(body.hits || []);
  } catch (error) {
    if (requestId === state.retrievalRequestId) {
      elements.retrievalResults.innerHTML = `<div class="retrieval-empty error"><span>!</span><strong>检索失败</strong><p>${escapeHtml(error.message)}</p></div>`;
    }
  } finally {
    if (requestId === state.retrievalRequestId) $('#retrieval-submit').disabled = false;
  }
}
function renderRetrievalResults(hits) {
  state.retrievalHits = hits;
  $('#retrieval-result-count').textContent = hits.length ? `Top ${hits.length}` : '';
  if (!hits.length) {
    elements.retrievalResults.innerHTML = '<div class="retrieval-empty"><span>⌕</span><strong>没有找到匹配内容</strong><p>试试其他关键词或范围。</p></div>';
    return;
  }
  elements.retrievalResults.innerHTML = hits.map((hit, index) => {
    const score = Number.isFinite(hit.score) ? hit.score.toFixed(3) : '';
    const summary = hit.abstract || hit.overview || hit.content_excerpt || '暂无摘要';
    return `<button class="retrieval-result" type="button" data-search-index="${index}">
      <span class="type-chip" data-type="${escapeHtml(hit.type)}">${escapeHtml(typeLabel(hit.type))}</span>
      <span class="retrieval-result-body"><strong>${escapeHtml(hit.id || '未命名节点')}</strong><small>${escapeHtml(hit.uri)}</small><span>${escapeHtml(summary)}</span></span>
      ${score ? `<span class="retrieval-score">${score}</span>` : ''}
    </button>`;
  }).join('');
}
function renderRetrievalDetail(hit) {
  if (!hit) return;
  $('#retrieval-detail-placeholder').hidden = true;
  $('#retrieval-detail-view').hidden = false;
  $('#retrieval-detail-type').textContent = typeLabel(hit.type);
  $('#retrieval-detail-type').dataset.type = hit.type || '';
  $('#retrieval-detail-title').textContent = hit.id || '未命名节点';
  $('#retrieval-detail-uri').textContent = hit.uri || '';
  $('#retrieval-detail-abstract').textContent = hit.abstract || '暂无摘要';
  $('#retrieval-detail-overview').textContent = displayValue(hit.overview, '暂无概览');
  $('#retrieval-detail-content').textContent = displayValue(hit.content_excerpt, '暂无正文');
  renderMetadataInto($('#retrieval-detail-metadata'), {
    score: hit.score,
    when: hit.when,
    who: hit.who,
    where: hit.where,
  });
  elements.retrievalDetail.scrollIntoView({ block: 'nearest' });
}
function clearRetrievalDetail() {
  state.retrievalHits = [];
  $('#retrieval-detail-view').hidden = true;
  $('#retrieval-detail-placeholder').hidden = false;
}
function resetRetrieval() {
  state.retrievalRequestId += 1;
  state.lastSearchTarget = null;
  clearRetrievalDetail();
  $('#retrieval-submit').disabled = false;
  $('#retrieval-result-count').textContent = '';
  elements.retrievalResults.innerHTML = '<div class="retrieval-empty"><span>⌕</span><strong>输入内容后开始检索</strong><p>结果将按语义相关度排序。</p></div>';
}
function switchView(view) {
  state.view = view;
  document.querySelectorAll('[data-view]').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === view);
    if (item.matches('.sidebar-link')) item.toggleAttribute('aria-current', item.dataset.view === view);
  });
  const dashboard = view === 'dashboard';
  const retrieval = view === 'retrieval';
  elements.dashboardView.hidden = !dashboard;
  elements.memoryView.hidden = dashboard || retrieval;
  elements.retrievalView.hidden = !retrieval;
  elements.pageTitle.textContent = dashboard ? '控制台' : retrieval ? '检索' : view === 'skill' ? 'Agent Skill' : 'ContextFS';
  const skill = view === 'skill';
  $('#tree-label').textContent = skill ? 'Agent Skill' : 'ContextFS';
  elements.tree.setAttribute('aria-label', skill ? 'Agent Skill 目录树' : 'ContextFS 目录树');
  elements.search.placeholder = skill ? '搜索已加载 Agent Skill、路径或摘要' : '搜索已加载节点、路径或摘要';
  $('#detail-placeholder-title').textContent = skill ? '选择一个 Agent Skill' : '选择一个 ContextFS 节点';
  $('#detail-placeholder-copy').textContent = skill
    ? '展开 Agent 目录后查看已生成或导入的 Skill。'
    : '展开目录后查看摘要、概览、正文与元数据。';
  if (dashboard) { renderDashboard(); return; }
  if (retrieval) { $('#retrieval-input').focus(); return; }
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
  const account = event.target.closest('[data-account-switch]');
  if (account) {
    state.accountId = account.dataset.accountSwitch;
    await loadTree();
    return;
  }
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
    state.selectedDirectoryKey = resource.closest('[data-node-key]')?.dataset.nodeKey || '';
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
$('#retrieval-form').addEventListener('submit', (event) => {
  event.preventDefault();
  runRetrieval();
});
$('#retrieval-filters-toggle').addEventListener('click', () => {
  const filters = $('#retrieval-filters');
  filters.hidden = !filters.hidden;
  $('#retrieval-filters-toggle').setAttribute('aria-expanded', String(!filters.hidden));
});
elements.retrievalResults.addEventListener('click', (event) => {
  const result = event.target.closest('[data-search-index]');
  if (!result) return;
  elements.retrievalResults.querySelectorAll('.retrieval-result').forEach((item) => item.classList.remove('active'));
  result.classList.add('active');
  renderRetrievalDetail(state.retrievalHits[Number(result.dataset.searchIndex)]);
});
elements.refresh.addEventListener('click', loadTree);
elements.revealCurrent.addEventListener('click', revealSelectedResource);
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
$('#settings-close').addEventListener('click', () => elements.settings.close());
$('#api-key-input').addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.isComposing) return;
  event.preventDefault();
  $('#settings-form').requestSubmit();
});
$('#settings-form').addEventListener('submit', (event) => {
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
