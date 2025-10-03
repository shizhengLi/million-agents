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

# 导入消息传播模型
from src.message_propagation.viral_propagation import ViralPropagationModel
from src.message_propagation.information_diffusion import InformationDiffusionModel
from src.message_propagation.influence_maximization import InfluenceMaximization
from src.message_propagation.propagation_tracker import PropagationTracker
from src.message_propagation.social_network import SocialNetwork as BaseSocialNetwork


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSocialNetwork(BaseSocialNetwork):
    """简单的社交网络实现类，用于Web API"""

    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, List[tuple]] = {}

    def add_agent(self, agent_id: str, agent_data: Optional[Dict[str, Any]] = None):
        """添加智能体到网络"""
        self.agents[agent_id] = agent_data or {}
        if agent_id not in self.connections:
            self.connections[agent_id] = []

    def add_connection(self, source_id: str, target_id: str, weight: float = 1.0):
        """添加连接关系"""
        if source_id in self.connections and target_id in self.agents:
            self.connections[source_id].append((target_id, weight))

    def get_neighbors(self, agent_id: str) -> List[str]:
        """获取智能体的邻居"""
        if agent_id in self.connections:
            return [neighbor for neighbor, _ in self.connections[agent_id]]
        return []

    def get_agent_count(self) -> int:
        """获取网络中智能体总数"""
        return len(self.agents)

    def get_all_agents(self) -> List[str]:
        """获取所有智能体ID"""
        return list(self.agents.keys())

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体信息"""
        return self.agents.get(agent_id)


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


# 消息传播相关数据模型
class PropagationRequest(BaseModel):
    """传播请求模型"""
    message: str = Field(..., min_length=1, max_length=1000, description="传播消息内容")
    seed_agents: List[str] = Field(..., min_length=1, description="种子智能体ID列表")
    model_type: str = Field(..., description="传播模型: viral, diffusion, hybrid")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="模型参数")
    max_steps: Optional[int] = Field(100, description="最大传播步数")


class PropagationResponse(BaseModel):
    """传播响应模型"""
    session_id: str
    status: str
    message: str
    model_type: str
    seed_agents: List[str]
    influenced_agents: List[str]
    propagation_steps: int
    statistics: Dict[str, Any]
    created_at: datetime


class InfluenceMaximizationRequest(BaseModel):
    """影响力最大化请求模型"""
    network_data: Optional[Dict[str, Any]] = Field(None, description="网络数据")
    seed_count: int = Field(..., ge=1, le=100, description="种子数量")
    algorithm: str = Field("greedy", description="算法: greedy, degree, celf")
    model_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="传播模型参数")


class InfluenceMaximizationResponse(BaseModel):
    """影响力最大化响应模型"""
    optimal_seeds: List[str]
    expected_influence: int
    algorithm_used: str
    computation_time: float
    network_stats: Dict[str, Any]


class NetworkMetricsResponse(BaseModel):
    """网络指标响应模型"""
    node_count: int
    edge_count: int
    average_degree: float
    clustering_coefficient: float
    average_path_length: float
    network_density: float
    connected_components: int
    largest_component_size: int


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


class MessagePropagationService:
    """消息传播服务类"""

    def __init__(self):
        # 延迟初始化传播模型，在需要时创建
        self.viral_model = None
        self.diffusion_model = None
        self.influence_maximizer = None
        self.propagation_tracker = None
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def _initialize_models(self, social_network: BaseSocialNetwork):
        """初始化传播模型"""
        if self.viral_model is None:
            self.viral_model = ViralPropagationModel(social_network)
        if self.diffusion_model is None:
            self.diffusion_model = InformationDiffusionModel(social_network)
        if self.influence_maximizer is None:
            self.influence_maximizer = InfluenceMaximization(social_network)
        if self.propagation_tracker is None:
            self.propagation_tracker = PropagationTracker(social_network)

    async def start_propagation(self, request: PropagationRequest) -> PropagationResponse:
        """启动消息传播"""
        import uuid

        session_id = str(uuid.uuid4())

        # 验证种子智能体是否存在
        agents = await agent_service.get_all_agents()
        agent_ids = [agent['id'] for agent in agents]

        invalid_seeds = [seed for seed in request.seed_agents if seed not in agent_ids]
        if invalid_seeds:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid seed agents: {invalid_seeds}"
            )

        # 创建社交网络
        social_network = SimpleSocialNetwork()
        for agent in agents:
            social_network.add_agent(agent['id'])

        # 添加连接关系
        for agent in agents:
            connections = await social_network_service.get_agent_connections(agent['id'])
            for conn in connections:
                social_network.add_connection(agent['id'], conn['target_id'], conn.get('strength', 1.0))

        # 初始化传播模型
        self._initialize_models(social_network)

        # 选择传播模型
        if request.model_type == "viral":
            model = self.viral_model
        elif request.model_type == "diffusion":
            model = self.diffusion_model
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Must be 'viral' or 'diffusion'"
            )

        # 设置传播参数
        parameters = request.parameters.copy()
        if request.model_type == "viral":
            model.infection_probability = parameters.get('infection_probability', 0.1)
            model.recovery_probability = parameters.get('recovery_probability', 0.05)
        elif request.model_type == "diffusion":
            model.adoption_probability = parameters.get('adoption_probability', 0.1)
            model.threshold = parameters.get('threshold', 0.3)

        # 设置种子智能体
        if request.model_type == "viral":
            model.set_initial_infected(request.seed_agents)
        else:
            model.set_initial_adopters(request.seed_agents)

        # 执行传播
        influenced_agents = []
        propagation_steps = 0

        try:
            if request.model_type == "viral":
                for step in range(request.max_steps):
                    model.propagate_step()
                    propagation_steps = step + 1
                    current_infected = list(model.infected_agents)
                    if len(current_infected) == len(influenced_agents):
                        break  # 传播结束
                    influenced_agents = current_infected
            else:
                for step in range(request.max_steps):
                    model.diffuse_step()
                    propagation_steps = step + 1
                    current_adopted = list(model.adopted_agents)
                    if len(current_adopted) == len(influenced_agents):
                        break  # 传播结束
                    influenced_agents = current_adopted

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Propagation simulation failed: {str(e)}"
            )

        # 计算传播统计
        statistics = {
            "total_influenced": len(influenced_agents),
            "seed_count": len(request.seed_agents),
            "propagation_ratio": len(influenced_agents) / len(agents) if agents else 0,
            "propagation_steps": propagation_steps,
            "model_parameters": parameters
        }

        # 保存会话信息
        session_data = {
            "session_id": session_id,
            "request": request.dict(),
            "result": {
                "influenced_agents": influenced_agents,
                "propagation_steps": propagation_steps,
                "statistics": statistics
            },
            "created_at": datetime.utcnow()
        }
        self.active_sessions[session_id] = session_data

        return PropagationResponse(
            session_id=session_id,
            status="completed",
            message="Propagation simulation completed successfully",
            model_type=request.model_type,
            seed_agents=request.seed_agents,
            influenced_agents=influenced_agents,
            propagation_steps=propagation_steps,
            statistics=statistics,
            created_at=datetime.utcnow()
        )

    async def get_propagation_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取传播结果"""
        return self.active_sessions.get(session_id)

    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """获取活跃传播会话"""
        return list(self.active_sessions.values())

    async def calculate_influence_maximization(self, request: InfluenceMaximizationRequest) -> InfluenceMaximizationResponse:
        """计算影响力最大化"""
        import time

        start_time = time.time()

        # 获取网络数据
        if request.network_data:
            network_data = request.network_data
        else:
            network_data = await social_network_service.get_network_data(limit=1000)

        # 创建社交网络
        social_network = SimpleSocialNetwork()

        # 添加节点
        for node in network_data.get('nodes', []):
            social_network.add_agent(node['id'])

        # 添加边
        for edge in network_data.get('edges', []):
            social_network.add_connection(edge['source'], edge['target'], edge.get('weight', 1.0))

        # 初始化模型
        self._initialize_models(social_network)

        # 确保影响力最大化器已正确初始化
        if self.influence_maximizer is None:
            raise HTTPException(
                status_code=500,
                detail="Influence maximization model failed to initialize"
            )

        # 设置模型参数
        model_params = request.model_parameters.copy()
        self.influence_maximizer.infection_probability = model_params.get('infection_probability', 0.1)
        self.influence_maximizer.simulation_rounds = model_params.get('simulation_rounds', 100)

        # 选择算法
        if request.algorithm == "greedy":
            optimal_seeds = self.influence_maximizer.greedy_algorithm(request.seed_count)
        elif request.algorithm == "degree":
            optimal_seeds = self.influence_maximizer.degree_heuristic(request.seed_count)
        elif request.algorithm == "celf":
            optimal_seeds = self.influence_maximizer.celf_algorithm(request.seed_count)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid algorithm. Must be 'greedy', 'degree', or 'celf'"
            )

        # 估算影响力
        expected_influence = self.influence_maximizer.estimate_influence(optimal_seeds)

        computation_time = time.time() - start_time

        # 计算网络统计
        network_stats = {
            "node_count": len(network_data.get('nodes', [])),
            "edge_count": len(network_data.get('edges', [])),
            "average_degree": sum(len(node.get('connections', [])) for node in network_data.get('nodes', [])) / len(network_data.get('nodes', [1])) if network_data.get('nodes') else 0
        }

        return InfluenceMaximizationResponse(
            optimal_seeds=optimal_seeds,
            expected_influence=expected_influence,
            algorithm_used=request.algorithm,
            computation_time=computation_time,
            network_stats=network_stats
        )

    async def get_network_metrics(self) -> NetworkMetricsResponse:
        """获取网络指标"""
        network_data = await social_network_service.get_network_data(limit=1000)
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])

        # 计算基本指标
        node_count = len(nodes)
        edge_count = len(edges)

        # 计算平均度数
        if node_count > 0:
            average_degree = (2 * edge_count) / node_count
        else:
            average_degree = 0

        # 简化的聚类系数计算
        clustering_coefficient = 0.3  # 模拟值

        # 简化的平均路径长度
        average_path_length = 3.5  # 模拟值

        # 网络密度
        max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0
        network_density = edge_count / max_edges if max_edges > 0 else 0

        # 连通分量
        connected_components = 1  # 简化假设
        largest_component_size = node_count  # 简化假设

        return NetworkMetricsResponse(
            node_count=node_count,
            edge_count=edge_count,
            average_degree=average_degree,
            clustering_coefficient=clustering_coefficient,
            average_path_length=average_path_length,
            network_density=network_density,
            connected_components=connected_components,
            largest_component_size=largest_component_size
        )


# 创建服务实例
agent_service = AgentService()
social_network_service = SocialNetworkService()
connection_manager = ConnectionManager()

# 延迟创建消息传播服务实例
def get_message_propagation_service():
    """获取消息传播服务实例"""
    if not hasattr(get_message_propagation_service, '_instance'):
        get_message_propagation_service._instance = MessagePropagationService()
    return get_message_propagation_service._instance


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
        """根路径，返回传播仪表板"""
        try:
            with open("src/web_interface/static/propagation_dashboard.html", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Million Agents Web Interface</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
            </head>
            <body>
                <div style="text-align: center; margin-top: 100px;">
                    <h1>Million Agents Web Interface</h1>
                    <p>正在加载仪表板...</p>
                    <p><a href="/docs">查看API文档</a></p>
                </div>
            </body>
            </html>
            """

    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        """返回传播仪表板页面"""
        try:
            with open("src/web_interface/static/propagation_dashboard.html", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "<h1>仪表板页面未找到</h1>"

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

    # Message Propagation APIs
    @app.post("/api/propagation/start", response_model=PropagationResponse)
    async def start_propagation(request: PropagationRequest):
        """启动消息传播模拟"""
        try:
            message_propagation_service = get_message_propagation_service()
            result = await message_propagation_service.start_propagation(request)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Propagation simulation failed: {str(e)}")

    @app.get("/api/propagation/session/{session_id}")
    async def get_propagation_session(session_id: str):
        """获取传播会话结果"""
        message_propagation_service = get_message_propagation_service()
        session = await message_propagation_service.get_propagation_result(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Propagation session not found")
        return session

    @app.get("/api/propagation/sessions")
    async def get_active_propagation_sessions():
        """获取所有活跃的传播会话"""
        message_propagation_service = get_message_propagation_service()
        sessions = await message_propagation_service.get_active_sessions()
        return {"sessions": sessions}

    @app.post("/api/influence-maximization", response_model=InfluenceMaximizationResponse)
    async def calculate_influence_maximization(request: InfluenceMaximizationRequest):
        """计算影响力最大化"""
        try:
            message_propagation_service = get_message_propagation_service()
            result = await message_propagation_service.calculate_influence_maximization(request)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Influence maximization calculation failed: {str(e)}")

    @app.get("/api/network/metrics", response_model=NetworkMetricsResponse)
    async def get_network_metrics():
        """获取网络拓扑指标"""
        try:
            message_propagation_service = get_message_propagation_service()
            metrics = await message_propagation_service.get_network_metrics()
            return metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to calculate network metrics: {str(e)}")

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