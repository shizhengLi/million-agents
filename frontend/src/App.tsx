import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  DashboardOutlined,
  ApiOutlined,
  TeamOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Propagation from './pages/Propagation'
import Agents from './pages/Agents'
import Network from './pages/Network'
import Influence from './pages/Influence'

const { Header, Content, Sider } = Layout

function App() {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: '仪表板',
    },
    {
      key: '/propagation',
      icon: <ThunderboltOutlined />,
      label: '传播模拟',
    },
    {
      key: '/network',
      icon: <ApiOutlined />,
      label: '网络可视化',
    },
    {
      key: '/agents',
      icon: <TeamOutlined />,
      label: '智能体管理',
    },
    {
      key: '/influence',
      icon: <BarChartOutlined />,
      label: '影响力分析',
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', padding: '0 24px', color: '#fff', fontSize: 20 }}>
        百万级智能体社交网络管理平台
      </Header>
      <Layout>
        <Sider width={200} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            style={{ height: '100%', borderRight: 0 }}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
          />
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 280, borderRadius: 8 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/propagation" element={<Propagation />} />
              <Route path="/network" element={<Network />} />
              <Route path="/agents" element={<Agents />} />
              <Route path="/influence" element={<Influence />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Layout>
  )
}

export default App
