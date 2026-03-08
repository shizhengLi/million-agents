import { useEffect, useState } from 'react'
import { Card, Form, InputNumber, Select, Button, Result, Tag, Space, Divider, message, Row, Col } from 'antd'
import { ThunderboltOutlined, ReloadOutlined } from '@ant-design/icons'
import { agentApi, influenceApi, Agent } from '../services/api'

export default function Influence() {
  const [loading, setLoading] = useState(false)
  const [agents, setAgents] = useState<Agent[]>([])
  const [result, setResult] = useState<{ optimal_seeds: string[]; expected_influence: number } | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    loadAgents()
  }, [])

  const loadAgents = async () => {
    try {
      const data = await agentApi.getAll()
      setAgents(data)
    } catch (error) {
      message.error('加载智能体失败')
    }
  }

  const handleCalculate = async (values: any) => {
    setLoading(true)
    try {
      const data = await influenceApi.calculate({
        seed_count: values.seed_count,
        algorithm: values.algorithm,
      })
      setResult(data)
      message.success('计算完成')
    } catch (error) {
      message.error('计算失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <h1 className="page-title">影响力分析</h1>

      <Card title="影响力最大化计算">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCalculate}
          initialValues={{
            seed_count: 3,
            algorithm: 'greedy',
          }}
        >
          <Form.Item
            name="seed_count"
            label="种子数量"
            rules={[{ required: true, message: '请输入种子数量' }]}
          >
            <InputNumber min={1} max={Math.min(20, agents.length)} style={{ width: 200 }} />
          </Form.Item>

          <Form.Item
            name="algorithm"
            label="算法"
            rules={[{ required: true, message: '请选择算法' }]}
          >
            <Select style={{ width: 200 }}>
              <Select.Option value="greedy">
                贪心算法 (Greedy) - 精确但较慢
              </Select.Option>
              <Select.Option value="degree">
                度启发式 (Degree) - 快速近似
              </Select.Option>
              <Select.Option value="celf">
                CELF算法 - 优化的贪心
              </Select.Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<ThunderboltOutlined />}
                loading={loading}
              >
                计算最优种子
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  form.resetFields()
                  setResult(null)
                }}
              >
                重置
              </Button>
            </Space>
          </Form.Item>
        </Form>

        {result && (
          <>
            <Divider />
            <Result
              status="success"
              title="计算完成"
              subTitle={`预期影响力: ${result.expected_influence}`}
              extra={
                <div style={{ textAlign: 'left', maxWidth: 400, margin: '0 auto' }}>
                  <h4 style={{ marginBottom: 12 }}>最优种子智能体:</h4>
                  <Space wrap>
                    {result.optimal_seeds.map((seed, index) => (
                      <Tag key={seed} color="blue" style={{ fontSize: 14, padding: '4px 12px' }}>
                        #{index + 1} {seed}
                      </Tag>
                    ))}
                  </Space>
                </div>
              }
            />
          </>
        )}
      </Card>

      <Card title="算法说明" style={{ marginTop: 16 }}>
        <h4>影响力最大化 (Influence Maximization)</h4>
        <p style={{ color: '#666', lineHeight: 1.8 }}>
          影响力最大化问题是社交网络分析中的经典问题，目标是在网络中找到一小部分有影响力的节点（种子），
          使它们能够最大化信息的传播范围。
        </p>
        <Divider />
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <Card size="small" title="贪心算法 (Greedy)">
              <p style={{ fontSize: 12, color: '#666' }}>
                每次选择能够带来最大边际增益的节点。理论上保证至少达到最优解的63%，
                但计算复杂度较高。
              </p>
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card size="small" title="度启发式 (Degree)">
              <p style={{ fontSize: 12, color: '#666' }}>
                选择度（连接数）最高的节点作为种子。简单快速，但精确度不如贪心算法。
              </p>
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card size="small" title="CELF算法">
              <p style={{ fontSize: 12, color: '#666' }}>
                利用次模特性进行优化，大幅减少边际增益的计算次数，比贪心算法快几个数量级。
              </p>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  )
}
