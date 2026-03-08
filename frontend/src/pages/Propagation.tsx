import { useEffect, useState } from 'react'
import { Form, Card, Input, Select, Slider, Button, Table, Tag, Space, message, Spin, Alert, Row, Col } from 'antd'
import { PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import { agentApi, propagationApi, Agent, PropagationResponse } from '../services/api'

const { TextArea } = Input

export default function Propagation() {
  const [loading, setLoading] = useState(false)
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgents, setSelectedAgents] = useState<string[]>([])
  const [history, setHistory] = useState<any[]>([])
  const [result, setResult] = useState<PropagationResponse | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    loadAgents()
    loadHistory()
  }, [])

  const loadAgents = async () => {
    try {
      const data = await agentApi.getAll()
      setAgents(data)
    } catch (error) {
      message.error('加载智能体失败')
    }
  }

  const loadHistory = async () => {
    try {
      const data = await propagationApi.getHistory()
      setHistory(data.sessions || [])
    } catch (error) {
      console.error('Failed to load history:', error)
    }
  }

  const handleSubmit = async (values: any) => {
    if (selectedAgents.length === 0) {
      message.warning('请选择至少一个种子智能体')
      return
    }

    setLoading(true)
    try {
      const response = await propagationApi.start({
        message: values.message,
        seed_agents: selectedAgents,
        model_type: values.model_type,
        parameters: {
          infection_probability: values.infection_probability,
          recovery_probability: values.recovery_probability,
          max_steps: values.max_steps,
        },
      })
      setResult(response)
      message.success('传播模拟启动成功')
      loadHistory()
    } catch (error: any) {
      message.error(error.response?.data?.detail || '启动失败')
    } finally {
      setLoading(false)
    }
  }

  const agentOptions = agents.map(agent => ({
    label: `${agent.name} (${agent.id})`,
    value: agent.id,
  }))

  const historyColumns = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
      render: (id: string) => id.substring(0, 8) + '...',
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
    },
    {
      title: '模型类型',
      dataIndex: 'model_type',
      key: 'model_type',
      render: (type: string) => <Tag color={type === 'viral' ? 'blue' : 'green'}>{type}</Tag>,
    },
    {
      title: '种子数量',
      dataIndex: 'seed_count',
      key: 'seed_count',
    },
    {
      title: '影响数量',
      dataIndex: 'total_influenced',
      key: 'total_influenced',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'completed' ? 'success' : 'processing'}>{status}</Tag>
      ),
    },
  ]

  return (
    <div className="page-container">
      <h1 className="page-title">传播模拟</h1>

      <Spin spinning={loading}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Card title="模拟参数设置" className="form-container">
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmit}
                initialValues={{
                  model_type: 'viral',
                  infection_probability: 0.2,
                  recovery_probability: 0.1,
                  max_steps: 50,
                }}
              >
                <Form.Item
                  name="message"
                  label="传播消息"
                  rules={[{ required: true, message: '请输入传播消息' }]}
                >
                  <TextArea rows={3} placeholder="输入要传播的消息内容" />
                </Form.Item>

                <Form.Item name="model_type" label="传播模型">
                  <Select>
                    <Select.Option value="viral">病毒式传播模型</Select.Option>
                    <Select.Option value="diffusion">信息扩散模型</Select.Option>
                  </Select>
                </Form.Item>

                <Form.Item name="infection_probability" label="感染概率">
                  <Slider min={0} max={1} step={0.05} />
                </Form.Item>

                <Form.Item name="recovery_probability" label="恢复概率">
                  <Slider min={0} max={1} step={0.05} />
                </Form.Item>

                <Form.Item name="max_steps" label="最大传播步数">
                  <Slider min={10} max={100} />
                </Form.Item>

                <Form.Item label="种子智能体">
                  <Select
                    mode="multiple"
                    placeholder="选择种子智能体"
                    options={agentOptions}
                    value={selectedAgents}
                    onChange={setSelectedAgents}
                    style={{ width: '100%' }}
                  />
                  <div style={{ marginTop: 8, color: '#888' }}>
                    已选择: {selectedAgents.length} 个智能体
                  </div>
                </Form.Item>

                <Form.Item>
                  <Space>
                    <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />}>
                      启动模拟
                    </Button>
                    <Button icon={<ReloadOutlined />} onClick={() => {
                      form.resetFields()
                      setSelectedAgents([])
                      setResult(null)
                    }}>
                      重置
                    </Button>
                  </Space>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            {result && (
              <Card title="模拟结果" style={{ marginBottom: 16 }}>
                <Alert
                  message="模拟完成"
                  description={`会话ID: ${result.session_id}`}
                  type="success"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
                <p><strong>模型类型:</strong> {result.model_type}</p>
                <p><strong>种子数量:</strong> {result.seed_count}</p>
                <p><strong>总智能体数:</strong> {result.total_agents}</p>
                <p><strong>受影响数量:</strong> {result.statistics.total_influenced}</p>
                <p><strong>传播步数:</strong> {result.statistics.propagation_steps}</p>
                <p><strong>传播率:</strong> {(result.statistics.propagation_rate * 100).toFixed(2)}%</p>
              </Card>
            )}

            <Card title="模拟历史">
              <Table
                dataSource={history}
                columns={historyColumns}
                rowKey="session_id"
                pagination={{ pageSize: 5 }}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      </Spin>
    </div>
  )
}
