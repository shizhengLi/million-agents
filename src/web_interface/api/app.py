"""
FastAPI Application
百万级智能体社交网络Web管理界面
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json
import logging

# 导入服务层
# from src.agents.agent_manager import AgentManager
# from src.social_network.social_graph import SocialGraph
# from src.reputation_system.reputation_engine import ReputationEngine
from .mock_services import MockAgentService, MockSocialNetworkService


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentCreateRequest(BaseModel):
    """创建智能体请求模型"""
    name: str = Field(..., min_length=1, max_length=100, description="智能体名称")
    type: str = Field(..., description="智能体类型: social, content, hybrid")
    description: Optional[str] = Field(None, max_length=500, description="智能体描述")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="智能体配置")


class AgentUpdateRequest(BaseModel):
    """更新智能体请求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="智能体名称")
    description: Optional[str] = Field(None, max_length=500, description="智能体描述")
    status: Optional[str] = Field(None, description="智能体状态: active, inactive, suspended")
    config: Optional[Dict[str, Any]] = Field(None, description="智能体配置")


class AgentResponse(BaseModel):
    """智能体响应模型"""
    id: str
    name: str
    type: str
    status: str
    reputation_score: float
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_active: Optional[datetime] = None


class NetworkNode(BaseModel):
    """网络节点模型"""
    id: str
    name: str
    group: str
    reputation_score: float
    status: str


class NetworkEdge(BaseModel):
    """网络边模型"""
    source: str
    target: str
    weight: float
    relationship_type: str
    last_interaction: Optional[datetime] = None


class NetworkResponse(BaseModel):
    """社交网络响应模型"""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    metrics: Dict[str, Any]


class ConnectionResponse(BaseModel):
    """连接关系响应模型"""
    target_id: str
    target_name: str
    relationship_type: str
    strength: float
    last_interaction: Optional[datetime] = None


class SystemStatsResponse(BaseModel):
    """系统统计响应模型"""
    agents: Dict[str, Any]
    social_network: Dict[str, Any]
    reputation_system: Dict[str, Any]
    timestamp: datetime


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """广播消息"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # 连接已断开，移除
                self.active_connections.remove(connection)


class AgentService:
    """智能体服务类"""

    def __init__(self):
        self.mock_service = MockAgentService()

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """获取所有智能体"""
        return await self.mock_service.get_all_agents()

    async def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取智能体"""
        return await self.mock_service.get_agent_by_id(agent_id)

    async def create_agent(self, agent_data: AgentCreateRequest) -> Dict[str, Any]:
        """创建智能体"""
        # 验证智能体类型
        valid_types = ["social", "content", "hybrid"]
        if agent_data.type not in valid_types:
            raise ValueError(f"Invalid agent type. Must be one of: {valid_types}")

        return await self.mock_service.create_agent(agent_data.dict())

    async def update_agent(self, agent_id: str, update_data: AgentUpdateRequest) -> Optional[Dict[str, Any]]:
        """更新智能体"""
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        return await self.mock_service.update_agent(agent_id, update_dict)

    async def delete_agent(self, agent_id: str) -> bool:
        """删除智能体"""
        return await self.mock_service.delete_agent(agent_id)

    async def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计"""
        return await self.mock_service.get_agent_stats()

    async def get_reputation_metrics(self) -> Dict[str, Any]:
        """获取声誉指标"""
        return await self.mock_service.get_reputation_metrics()


class SocialNetworkService:
    """社交网络服务类"""

    def __init__(self):
        self.mock_service = MockSocialNetworkService()

    async def get_network_data(self, limit: int = 100) -> Dict[str, Any]:
        """获取网络数据"""
        return await self.mock_service.get_network_data(limit)

    async def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """获取智能体连接"""
        return await self.mock_service.get_agent_connections(agent_id)

    async def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        return await self.mock_service.get_network_stats()


# 创建服务实例
agent_service = AgentService()
social_network_service = SocialNetworkService()
connection_manager = ConnectionManager()


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="Million Agents Web Interface",
        description="百万级智能体社交网络管理界面",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应该限制为特定域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件
    app.mount("/static", StaticFiles(directory="src/web_interface/static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        """根路径，返回主页面"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Million Agents Web Interface</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="/static/css/style.css">
        </head>
        <body>
            <div class="container">
                <h1>Million Agents Web Interface</h1>
                <p>Welcome to the Million-scale Agent Social Network Management Interface</p>
                <nav>
                    <ul>
                        <li><a href="/agents">Agent Management</a></li>
                        <li><a href="/network">Network Visualization</a></li>
                        <li><a href="/metrics">System Metrics</a></li>
                        <li><a href="/docs">API Documentation</a></li>
                    </ul>
                </nav>
            </div>
            <script src="/static/js/main.js"></script>
        </body>
        </html>
        """

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "version": "1.0.0"
        }

    # Agent Management APIs
    @app.get("/api/agents", response_model=List[AgentResponse])
    async def get_agents():
        """获取所有智能体"""
        agents = await agent_service.get_all_agents()
        return agents

    @app.get("/api/agents/{agent_id}", response_model=AgentResponse)
    async def get_agent(agent_id: str):
        """获取指定智能体"""
        agent = await agent_service.get_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent

    @app.post("/api/agents", response_model=AgentResponse, status_code=201)
    async def create_agent(agent_data: AgentCreateRequest):
        """创建新智能体"""
        try:
            agent = await agent_service.create_agent(agent_data)
            return agent
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.put("/api/agents/{agent_id}", response_model=AgentResponse)
    async def update_agent(agent_id: str, update_data: AgentUpdateRequest):
        """更新智能体"""
        agent = await agent_service.update_agent(agent_id, update_data)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent

    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """删除智能体"""
        success = await agent_service.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"message": "Agent deleted successfully"}

    @app.get("/api/agents/{agent_id}/connections", response_model=List[ConnectionResponse])
    async def get_agent_connections(agent_id: str):
        """获取智能体连接"""
        connections = await social_network_service.get_agent_connections(agent_id)
        return connections

    # Social Network APIs
    @app.get("/api/social-network", response_model=NetworkResponse)
    async def get_social_network(limit: int = 100):
        """获取社交网络数据"""
        network_data = await social_network_service.get_network_data(limit)
        return network_data

    # Metrics APIs
    @app.get("/api/metrics/reputation")
    async def get_reputation_metrics():
        """获取声誉指标"""
        metrics = await agent_service.get_reputation_metrics()
        return metrics

    @app.get("/api/stats/system", response_model=SystemStatsResponse)
    async def get_system_stats():
        """获取系统统计"""
        agent_stats = await agent_service.get_agent_stats()
        network_stats = await social_network_service.get_network_stats()
        reputation_stats = await agent_service.get_reputation_metrics()

        return {
            "agents": agent_stats,
            "social_network": network_stats,
            "reputation_system": reputation_stats,
            "timestamp": datetime.utcnow()
        }

    # WebSocket
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket端点，用于实时数据更新"""
        await connection_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await connection_manager.send_personal_message("pong", websocket)
                elif data.startswith("subscribe:"):
                    # 处理订阅请求
                    channel = data.split(":", 1)[1]
                    await connection_manager.send_personal_message(
                        f"Subscribed to {channel}", websocket
                    )
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)