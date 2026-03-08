import { useEffect, useState } from 'react'
import { Table, Button, Space, Tag, Modal, Form, Input, Select, message, Popconfirm } from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined, ReloadOutlined } from '@ant-design/icons'
import { agentApi, Agent } from '../services/api'

export default function Agents() {
  const [loading, setLoading] = useState(true)
  const [agents, setAgents] = useState<Agent[]>([])
  const [modalVisible, setModalVisible] = useState(false)
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    loadAgents()
  }, [])

  const loadAgents = async () => {
    setLoading(true)
    try {
      const data = await agentApi.getAll()
      setAgents(data)
    } catch (error) {
      message.error('加载智能体失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreate = () => {
    setEditingAgent(null)
    form.resetFields()
    setModalVisible(true)
  }

  const handleEdit = (record: Agent) => {
    setEditingAgent(record)
    form.setFieldsValue({
      name: record.name,
      type: record.type,
      description: record.description,
    })
    setModalVisible(true)
  }

  const handleDelete = async (id: string) => {
    try {
      await agentApi.delete(id)
      message.success('删除成功')
      loadAgents()
    } catch (error) {
      message.error('删除失败')
    }
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      if (editingAgent) {
        await agentApi.update(editingAgent.id, values)
        message.success('更新成功')
      } else {
        await agentApi.create(values)
        message.success('创建成功')
      }
      setModalVisible(false)
      loadAgents()
    } catch (error) {
      message.error('操作失败')
    }
  }

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const colorMap: Record<string, string> = {
          social: 'blue',
          content: 'green',
          hybrid: 'purple',
          balanced: 'cyan',
          explorer: 'orange',
          builder: 'red',
          connector: 'gold',
        }
        return <Tag color={colorMap[type] || 'default'}>{type}</Tag>
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'success' : 'default'}>{status}</Tag>
      ),
    },
    {
      title: '声誉分数',
      dataIndex: 'reputation_score',
      key: 'reputation_score',
      sorter: (a: Agent, b: Agent) => a.reputation_score - b.reputation_score,
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Agent) => (
        <Space size="small">
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Popconfirm
            title="确认删除此智能体?"
            onConfirm={() => handleDelete(record.id)}
            okText="确认"
            cancelText="取消"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div className="page-container">
      <h1 className="page-title">智能体管理</h1>

      <Space style={{ marginBottom: 16 }}>
        <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
          添加智能体
        </Button>
        <Button icon={<ReloadOutlined />} onClick={loadAgents}>
          刷新
        </Button>
      </Space>

      <Table
        dataSource={agents}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10, showSizeChanger: true, showTotal: (total) => `共 ${total} 条` }}
      />

      <Modal
        title={editingAgent ? '编辑智能体' : '添加智能体'}
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
        width={500}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: '请输入智能体名称' }]}
          >
            <Input placeholder="输入智能体名称" />
          </Form.Item>

          <Form.Item
            name="type"
            label="类型"
            rules={[{ required: true, message: '请选择智能体类型' }]}
          >
            <Select placeholder="选择类型">
              <Select.Option value="social">社交型</Select.Option>
              <Select.Option value="content">内容型</Select.Option>
              <Select.Option value="hybrid">混合型</Select.Option>
              <Select.Option value="balanced">平衡型</Select.Option>
              <Select.Option value="explorer">探索型</Select.Option>
              <Select.Option value="builder">建设型</Select.Option>
              <Select.Option value="connector">连接型</Select.Option>
              <Select.Option value="leader">领导型</Select.Option>
              <Select.Option value="innovator">创新型</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="description" label="描述">
            <Input.TextArea rows={3} placeholder="输入智能体描述" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
