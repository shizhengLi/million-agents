import { useEffect, useState, useRef } from 'react'
import { Card, Spin, Select, Row, Col } from 'antd'
import { networkApi, agentApi, Agent } from '../services/api'
import { Network as VisNetwork, DataSet } from 'vis-network/standalone'

export default function Network() {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<VisNetwork | null>(null)
  const [loading, setLoading] = useState(true)
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [stats, setStats] = useState<{ total: number; connections: number }>({ total: 0, connections: 0 })

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    if (!loading && containerRef.current) {
      initNetwork()
    }
  }, [loading, agents])

  const loadData = async () => {
    try {
      const [agentData, networkData] = await Promise.all([
        agentApi.getAll(),
        networkApi.getData(200),
      ])
      setAgents(agentData)
      setStats({
        total: agentData.length,
        connections: networkData.connections.length,
      })
    } catch (error) {
      console.error('Failed to load network data:', error)
    } finally {
      setLoading(false)
    }
  }

  const initNetwork = () => {
    const container = containerRef.current
    if (!container) return

    const nodes = new DataSet<any>(
      agents.map(agent => ({
        id: agent.id,
        label: agent.name,
        title: `${agent.name}\n类型: ${agent.type}\n状态: ${agent.status}`,
        color: agent.status === 'active' ? '#52c41a' : '#d9d9d9',
        size: 20,
      }))
    )

    const edges: any[] = []
    try {
      networkApi.getData(500).then(data => {
        data.connections.forEach((conn: any) => {
          edges.push({
            from: conn.source_id,
            to: conn.target_id,
            width: conn.strength * 2,
            title: `连接强度: ${conn.strength.toFixed(2)}`,
          })
        })

        const networkData = { nodes, edges: new DataSet(edges) }
        const options = {
          nodes: {
            shape: 'dot',
            font: { size: 14, color: '#333' },
            borderWidth: 2,
          },
          edges: {
            color: { color: '#b4b4b4', highlight: '#1890ff' },
            smooth: true,
          },
          physics: {
            enabled: true,
            barnesHut: {
              gravitationalConstant: -2000,
              centralGravity: 0.3,
              springLength: 100,
            },
          },
          interaction: {
            hover: true,
            tooltipDelay: 200,
          },
        }

        if (networkRef.current) {
          networkRef.current.destroy()
        }
        networkRef.current = new VisNetwork(container, networkData as any, options)
      })
    } catch (error) {
      const networkData = { nodes, edges: new DataSet(edges) }
      const options = {
        nodes: {
          shape: 'dot',
          font: { size: 14, color: '#333' },
          borderWidth: 2,
        },
        edges: {
          color: { color: '#b4b4b4' },
        },
        physics: { enabled: true },
      }

      if (networkRef.current) {
        networkRef.current.destroy()
      }
      networkRef.current = new VisNetwork(container, networkData as any, options)
    }
  }

  const handleAgentSelect = async (agentId: string) => {
    setSelectedAgent(agentId)
    if (networkRef.current) {
      networkRef.current.selectNodes([agentId])
      networkRef.current.focus(agentId, { animation: true, scale: 1.2 })
    }
  }

  const agentOptions = agents.map(agent => ({
    label: `${agent.name} (${agent.id})`,
    value: agent.id,
  }))

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="page-container">
      <h1 className="page-title">网络可视化</h1>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={8}>
          <Card size="small" title="网络统计">
            <p><strong>智能体总数:</strong> {stats.total}</p>
            <p><strong>连接总数:</strong> {stats.connections}</p>
          </Card>
          <Card size="small" title="查找智能体" style={{ marginTop: 16 }}>
            <Select
              showSearch
              placeholder="搜索智能体"
              style={{ width: '100%' }}
              options={agentOptions}
              value={selectedAgent}
              onChange={handleAgentSelect}
              allowClear
            />
          </Card>
        </Col>
        <Col xs={24} sm={16}>
          <Card bodyStyle={{ padding: 0 }}>
            <div ref={containerRef} className="network-container" style={{ height: 500 }} />
          </Card>
        </Col>
      </Row>

      <Card title="操作说明" size="small" style={{ marginTop: 16 }}>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          <li>拖拽画布移动视图</li>
          <li>滚轮缩放</li>
          <li>点击节点查看详情</li>
          <li>双击节点聚焦</li>
        </ul>
      </Card>
    </div>
  )
}
