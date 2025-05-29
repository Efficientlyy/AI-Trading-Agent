// --- SYSTEM CONTROL API ---

export async function startSystem() {
  const res = await fetch('/system/start', { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start system');
  return res.json();
}

export async function stopSystem() {
  const res = await fetch('/system/stop', { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop system');
  return res.json();
}

export async function getSystemStatus() {
  const res = await fetch('/system/status');
  if (!res.ok) throw new Error('Failed to fetch system status');
  return res.json();
}

// --- AGENT CONTROL API ---

export async function getAgents() {
  const res = await fetch('/system/agents');
  if (!res.ok) throw new Error('Failed to fetch agents');
  return res.json();
}

export async function startAgent(agentId: string) {
  const res = await fetch(`/system/agents/${agentId}/start`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to start agent');
  return res.json();
}

export async function stopAgent(agentId: string) {
  const res = await fetch(`/system/agents/${agentId}/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop agent');
  return res.json();
}

export async function getAgent(agentId: string) {
  const res = await fetch(`/system/agents/${agentId}`);
  if (!res.ok) throw new Error('Failed to fetch agent info');
  return res.json();
}

// --- SESSIONS API ---

export async function getSessions() {
  const res = await fetch('/system/paper-trading/sessions');
  if (!res.ok) throw new Error('Failed to fetch sessions');
  return res.json();
}
