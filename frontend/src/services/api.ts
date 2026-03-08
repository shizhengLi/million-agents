import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export interface Agent {
  id: string
  name: string
  type: string
  status: string
  reputation_score: number
  description: string
  created_at: string
  updated_at: string
  last_active: string
}

export interface AgentStats {
  total_agents: number
  active_agents: number
  inactive_agents: number
  agent_types: Record<string, number>
}

export interface ReputationMetrics {
  total_agents: number
  active_agents: number
  average_reputation: number
  reputation_distribution: {
    high: number
    medium: number
    low: number
  }
  recent_changes: any[]
}

export interface NetworkData {
  connections: Array<{
    source_id: string
    target_id: string
    strength: number
  }>
}

export interface NetworkStats {
  total_connections: number
  average_degree: number
}

export interface PropagationRequest {
  message: string
  seed_agents: string[]
  model_type: 'viral' | 'diffusion'
  parameters: {
    infection_probability?: number
    recovery_probability?: number
    max_steps?: number
  }
}

export interface PropagationResponse {
  session_id: string
  status: string
  model_type: string
  seed_agents: string[]
  seed_count: number
  total_agents: number
  influenced_agents: string[]
  statistics: {
    total_influenced: number
    propagation_steps: number
    propagation_rate: number
  }
}

export interface InfluenceRequest {
  seed_count: number
  algorithm: 'greedy' | 'degree' | 'celf'
}

export interface InfluenceResponse {
  optimal_seeds: string[]
  expected_influence: number
}

export const agentApi = {
  getAll: () => api.get<Agent[]>('/agents/').then(res => res.data),
  getById: (id: string) => api.get<Agent>(`/agents/${id}`).then(res => res.data),
  create: (data: { name: string; type: string; description?: string }) =>
    api.post<Agent>('/agents/', data).then(res => res.data),
  update: (id: string, data: any) =>
    api.put<Agent>(`/agents/${id}`, data).then(res => res.data),
  delete: (id: string) => api.delete(`/agents/${id}`).then(res => res.data),
  getStats: () => api.get<AgentStats>('/agents/stats').then(res => res.data),
  getReputation: () => api.get<ReputationMetrics>('/agents/reputation').then(res => res.data),
}

export const networkApi = {
  getData: (limit = 100) => api.get<NetworkData>('/network/data', { params: { limit } }).then(res => res.data),
  getStats: () => api.get<NetworkStats>('/network/stats').then(res => res.data),
  getConnections: (agentId: string) =>
    api.get<Array<{ target_id: string; strength: number }>>(`/network/connections/${agentId}`).then(res => res.data),
}

export const propagationApi = {
  start: (data: PropagationRequest) =>
    api.post<PropagationResponse>('/propagation/start', data).then(res => res.data),
  getStatus: (sessionId: string) =>
    api.get(`/propagation/status/${sessionId}`).then(res => res.data),
  getHistory: () => api.get('/propagation/history').then(res => res.data),
}

export const influenceApi = {
  calculate: (data: InfluenceRequest) =>
    api.post<InfluenceResponse>('/influence/calculate', data).then(res => res.data),
}

export default api
