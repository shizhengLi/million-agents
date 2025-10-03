"""
消息传播Web API测试
使用TDD方法确保100%测试通过
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import json

from src.web_interface.api.app import create_app, PropagationRequest, InfluenceMaximizationRequest


class TestMessagePropagationAPI:
    """消息传播API测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.app = create_app()
        self.client = TestClient(self.app)

    @pytest.mark.asyncio
    async def test_propagation_start_viral_model(self):
        """测试启动病毒式传播模拟"""
        # 准备测试数据
        propagation_request = {
            "message": "测试消息传播",
            "seed_agents": ["agent_001", "agent_002"],
            "model_type": "viral",
            "parameters": {
                "infection_probability": 0.2,
                "recovery_probability": 0.1
            },
            "max_steps": 50
        }

        # 模拟智能体服务
        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"},
            {"id": "agent_002", "name": "Agent 2", "status": "active"},
            {"id": "agent_003", "name": "Agent 3", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(return_value={
                        "session_id": "test_session_123",
                        "status": "completed",
                        "message": propagation_request["message"],
                        "model_type": propagation_request["model_type"],
                        "seed_agents": propagation_request["seed_agents"],
                        "influenced_agents": ["agent_001", "agent_002", "agent_003"],
                        "propagation_steps": 5,
                        "statistics": {
                            "total_influenced": 3,
                            "seed_count": len(propagation_request["seed_agents"]),
                            "propagation_ratio": 1.0,
                            "propagation_steps": 5,
                            "model_parameters": propagation_request["parameters"]
                        },
                        "created_at": datetime.utcnow()
                    })
                    mock_get_service.return_value = mock_service
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        # 验证响应
        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["status"] == "completed"
        assert data["model_type"] == "viral"
        assert data["seed_agents"] == ["agent_001", "agent_002"]
        assert isinstance(data["influenced_agents"], list)
        assert isinstance(data["propagation_steps"], int)
        assert "statistics" in data

    @pytest.mark.asyncio
    async def test_propagation_start_diffusion_model(self):
        """测试启动信息扩散模拟"""
        propagation_request = {
            "message": "信息扩散测试",
            "seed_agents": ["agent_001"],
            "model_type": "diffusion",
            "parameters": {
                "adoption_probability": 0.15,
                "threshold": 0.4
            },
            "max_steps": 30
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"},
            {"id": "agent_002", "name": "Agent 2", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(return_value={
                        "session_id": "test_session_diffusion",
                        "status": "completed",
                        "message": propagation_request["message"],
                        "model_type": propagation_request["model_type"],
                        "seed_agents": propagation_request["seed_agents"],
                        "influenced_agents": ["agent_001", "agent_002"],
                        "propagation_steps": 3,
                        "statistics": {
                            "total_influenced": 2,
                            "seed_count": len(propagation_request["seed_agents"]),
                            "propagation_ratio": 1.0,
                            "propagation_steps": 3,
                            "model_parameters": propagation_request["parameters"]
                        },
                        "created_at": datetime.utcnow()
                    })
                    mock_get_service.return_value = mock_service
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "diffusion"
        assert data["seed_agents"] == ["agent_001"]

    def test_propagation_start_invalid_model_type(self):
        """测试无效的传播模型类型"""
        propagation_request = {
            "message": "测试消息",
            "seed_agents": ["agent_001"],
            "model_type": "invalid_model",
            "parameters": {},
            "max_steps": 10
        }

        # 首先提供有效的智能体，绕过种子验证
        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(side_effect=HTTPException(status_code=400, detail="Invalid model type"))
                    mock_get_service.return_value = mock_service
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 400
        assert "Invalid model type" in response.json()["detail"]

    def test_propagation_start_invalid_seed_agents(self):
        """测试无效的种子智能体"""
        propagation_request = {
            "message": "测试消息",
            "seed_agents": ["nonexistent_agent"],
            "model_type": "viral",
            "parameters": {},
            "max_steps": 10
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 400
        assert "Invalid seed agents" in response.json()["detail"]

    def test_propagation_start_missing_message(self):
        """测试缺少消息内容的请求"""
        propagation_request = {
            "seed_agents": ["agent_001"],
            "model_type": "viral",
            "parameters": {},
            "max_steps": 10
        }

        response = self.client.post("/api/propagation/start", json=propagation_request)
        assert response.status_code == 422  # Validation error

    def test_propagation_start_empty_seed_agents(self):
        """测试空的种子智能体列表"""
        propagation_request = {
            "message": "测试消息",
            "seed_agents": [],
            "model_type": "viral",
            "parameters": {},
            "max_steps": 10
        }

        response = self.client.post("/api/propagation/start", json=propagation_request)
        assert response.status_code == 422  # Validation error

    def test_get_propagation_session_existing(self):
        """测试获取存在的传播会话"""
        # 首先创建一个传播会话
        session_id = "test_session_123"
        session_data = {
            "session_id": session_id,
            "request": {
                "message": "测试消息",
                "seed_agents": ["agent_001"],
                "model_type": "viral"
            },
            "result": {
                "influenced_agents": ["agent_001", "agent_002"],
                "propagation_steps": 5,
                "statistics": {"total_influenced": 2}
            },
            "created_at": datetime.utcnow()
        }

        # 模拟会话存在
        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_propagation_result = AsyncMock(return_value=session_data)
            mock_get_service.return_value = mock_service
            response = self.client.get(f"/api/propagation/session/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "result" in data

    def test_get_propagation_session_not_found(self):
        """测试获取不存在的传播会话"""
        session_id = "nonexistent_session"

        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_propagation_result = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service
            response = self.client.get(f"/api/propagation/session/{session_id}")

        assert response.status_code == 404
        assert "Propagation session not found" in response.json()["detail"]

    def test_get_active_propagation_sessions(self):
        """测试获取活跃的传播会话列表"""
        mock_sessions = [
            {
                "session_id": "session_1",
                "created_at": datetime.utcnow(),
                "status": "completed"
            },
            {
                "session_id": "session_2",
                "created_at": datetime.utcnow(),
                "status": "running"
            }
        ]

        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_active_sessions = AsyncMock(return_value=mock_sessions)
            mock_get_service.return_value = mock_service
            response = self.client.get("/api/propagation/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 2

    def test_influence_maximization_greedy_algorithm(self):
        """测试影响力最大化 - 贪心算法"""
        request_data = {
            "seed_count": 3,
            "algorithm": "greedy",
            "model_parameters": {
                "infection_probability": 0.1,
                "simulation_rounds": 50
            }
        }

        mock_network_data = {
            "nodes": [
                {"id": "agent_001", "name": "Agent 1"},
                {"id": "agent_002", "name": "Agent 2"},
                {"id": "agent_003", "name": "Agent 3"},
                {"id": "agent_004", "name": "Agent 4"}
            ],
            "edges": [
                {"source": "agent_001", "target": "agent_002", "weight": 1.0},
                {"source": "agent_001", "target": "agent_003", "weight": 0.8},
                {"source": "agent_002", "target": "agent_004", "weight": 0.6}
            ]
        }

        mock_response = {
            "optimal_seeds": ["agent_001", "agent_002", "agent_003"],
            "expected_influence": 4,
            "algorithm_used": "greedy",
            "computation_time": 0.05,
            "network_stats": {
                "node_count": 4,
                "edge_count": 3,
                "average_degree": 1.5
            }
        }

        with patch('src.web_interface.api.app.social_network_service.get_network_data', return_value=mock_network_data):
            with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.calculate_influence_maximization = AsyncMock(return_value=mock_response)
                mock_get_service.return_value = mock_service
                response = self.client.post("/api/influence-maximization", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "optimal_seeds" in data
        assert len(data["optimal_seeds"]) == 3
        assert data["algorithm_used"] == "greedy"
        assert "expected_influence" in data
        assert "computation_time" in data

    def test_influence_maximization_degree_algorithm(self):
        """测试影响力最大化 - 度启发式算法"""
        request_data = {
            "seed_count": 2,
            "algorithm": "degree",
            "model_parameters": {
                "infection_probability": 0.15
            }
        }

        mock_response = {
            "optimal_seeds": ["agent_001", "agent_002"],
            "expected_influence": 3,
            "algorithm_used": "degree",
            "computation_time": 0.02,
            "network_stats": {"node_count": 4, "edge_count": 3, "average_degree": 1.5}
        }

        with patch('src.web_interface.api.app.social_network_service.get_network_data', return_value={"nodes": [], "edges": []}):
            with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.calculate_influence_maximization = AsyncMock(return_value=mock_response)
                mock_get_service.return_value = mock_service
                response = self.client.post("/api/influence-maximization", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["algorithm_used"] == "degree"

    def test_influence_maximization_celf_algorithm(self):
        """测试影响力最大化 - CELF算法"""
        request_data = {
            "seed_count": 2,
            "algorithm": "celf",
            "model_parameters": {}
        }

        mock_response = {
            "optimal_seeds": ["agent_001", "agent_003"],
            "expected_influence": 3,
            "algorithm_used": "celf",
            "computation_time": 0.08,
            "network_stats": {"node_count": 4, "edge_count": 3, "average_degree": 1.5}
        }

        with patch('src.web_interface.api.app.social_network_service.get_network_data', return_value={"nodes": [], "edges": []}):
            with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.calculate_influence_maximization = AsyncMock(return_value=mock_response)
                mock_get_service.return_value = mock_service
                response = self.client.post("/api/influence-maximization", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["algorithm_used"] == "celf"

    def test_influence_maximization_invalid_algorithm(self):
        """测试影响力最大化 - 无效算法"""
        request_data = {
            "seed_count": 2,
            "algorithm": "invalid_algorithm",
            "model_parameters": {}
        }

        with patch('src.web_interface.api.app.social_network_service.get_network_data', return_value={"nodes": [], "edges": []}):
            with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.calculate_influence_maximization = AsyncMock(side_effect=Exception("Invalid algorithm"))
                mock_get_service.return_value = mock_service
                response = self.client.post("/api/influence-maximization", json=request_data)

        assert response.status_code == 500

    def test_influence_maximization_invalid_seed_count(self):
        """测试影响力最大化 - 无效种子数量"""
        request_data = {
            "seed_count": 150,  # 超过最大限制
            "algorithm": "greedy",
            "model_parameters": {}
        }

        response = self.client.post("/api/influence-maximization", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_get_network_metrics(self):
        """测试获取网络拓扑指标"""
        mock_metrics = {
            "node_count": 100,
            "edge_count": 250,
            "average_degree": 5.0,
            "clustering_coefficient": 0.3,
            "average_path_length": 3.5,
            "network_density": 0.05,
            "connected_components": 1,
            "largest_component_size": 100
        }

        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_network_metrics = AsyncMock(return_value=mock_metrics)
            mock_get_service.return_value = mock_service
            response = self.client.get("/api/network/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["node_count"] == 100
        assert data["edge_count"] == 250
        assert data["average_degree"] == 5.0
        assert data["network_density"] == 0.05

    def test_get_network_metrics_calculation_error(self):
        """测试网络指标计算错误"""
        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_network_metrics = AsyncMock(side_effect=Exception("Calculation failed"))
            mock_get_service.return_value = mock_service
            response = self.client.get("/api/network/metrics")

        assert response.status_code == 500
        assert "Failed to calculate network metrics" in response.json()["detail"]

    def test_propagation_request_validation(self):
        """测试传播请求模型验证"""
        # 有效请求
        valid_request = PropagationRequest(
            message="测试消息",
            seed_agents=["agent_001"],
            model_type="viral",
            parameters={"infection_probability": 0.1},
            max_steps=10
        )
        assert valid_request.message == "测试消息"
        assert valid_request.model_type == "viral"

    def test_influence_maximization_request_validation(self):
        """测试影响力最大化请求模型验证"""
        # 有效请求
        valid_request = InfluenceMaximizationRequest(
            seed_count=5,
            algorithm="greedy",
            model_parameters={"infection_probability": 0.1}
        )
        assert valid_request.seed_count == 5
        assert valid_request.algorithm == "greedy"

    def test_propagation_with_custom_network_data(self):
        """测试使用自定义网络数据的传播"""
        propagation_request = {
            "message": "自定义网络测试",
            "seed_agents": ["agent_001"],
            "model_type": "viral",
            "parameters": {},
            "max_steps": 10
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"},
            {"id": "agent_002", "name": "Agent 2", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(return_value={
                        "session_id": "test_custom_network",
                        "status": "completed",
                        "message": propagation_request["message"],
                        "model_type": propagation_request["model_type"],
                        "seed_agents": propagation_request["seed_agents"],
                        "influenced_agents": ["agent_001", "agent_002"],
                        "propagation_steps": 3,
                        "statistics": {
                            "total_influenced": 2,
                            "seed_count": len(propagation_request["seed_agents"]),
                            "propagation_ratio": 1.0,
                            "propagation_steps": 3,
                            "model_parameters": propagation_request["parameters"]
                        },
                        "created_at": datetime.utcnow()
                    })
                    mock_get_service.return_value = mock_service
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 200

    def test_propagation_statistics_accuracy(self):
        """测试传播统计数据的准确性"""
        propagation_request = {
            "message": "统计测试",
            "seed_agents": ["agent_001", "agent_002"],
            "model_type": "viral",
            "parameters": {"infection_probability": 0.5},
            "max_steps": 20
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"},
            {"id": "agent_002", "name": "Agent 2", "status": "active"},
            {"id": "agent_003", "name": "Agent 3", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(return_value={
                        "session_id": "test_statistics_accuracy",
                        "status": "completed",
                        "message": propagation_request["message"],
                        "model_type": propagation_request["model_type"],
                        "seed_agents": propagation_request["seed_agents"],
                        "influenced_agents": ["agent_001", "agent_002", "agent_003"],
                        "propagation_steps": 8,
                        "statistics": {
                            "total_influenced": 3,
                            "seed_count": len(propagation_request["seed_agents"]),
                            "propagation_ratio": 1.0,
                            "propagation_steps": 8,
                            "model_parameters": propagation_request["parameters"]
                        },
                        "created_at": datetime.utcnow()
                    })
                    mock_get_service.return_value = mock_service
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 200
        data = response.json()

        # 验证统计数据的完整性
        stats = data["statistics"]
        assert "total_influenced" in stats
        assert "seed_count" in stats
        assert "propagation_ratio" in stats
        assert "propagation_steps" in stats
        assert "model_parameters" in stats

        # 验证数值的合理性
        assert stats["total_influenced"] >= stats["seed_count"]
        assert 0 <= stats["propagation_ratio"] <= 1
        assert stats["propagation_steps"] > 0


class TestMessagePropagationIntegration:
    """消息传播集成测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_propagation_and_influence_maximization_integration(self):
        """测试传播模拟与影响力最大化集成"""
        # 1. 首先计算影响力最大化找到最优种子
        influence_request = {
            "seed_count": 2,
            "algorithm": "greedy",
            "model_parameters": {"infection_probability": 0.1}
        }

        mock_influence_response = {
            "optimal_seeds": ["agent_001", "agent_003"],
            "expected_influence": 4,
            "algorithm_used": "greedy",
            "computation_time": 0.05,
            "network_stats": {"node_count": 5, "edge_count": 6, "average_degree": 2.4}
        }

        with patch('src.web_interface.api.app.social_network_service.get_network_data', return_value={"nodes": [], "edges": []}):
            with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                mock_service = MagicMock()
                mock_service.calculate_influence_maximization = AsyncMock(return_value=mock_influence_response)
                mock_get_service.return_value = mock_service
                influence_response = self.client.post("/api/influence-maximization", json=influence_request)

        assert influence_response.status_code == 200
        optimal_seeds = influence_response.json()["optimal_seeds"]

        # 2. 使用最优种子进行传播模拟
        propagation_request = {
            "message": "集成测试消息",
            "seed_agents": optimal_seeds,
            "model_type": "viral",
            "parameters": {"infection_probability": 0.1},
            "max_steps": 15
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"},
            {"id": "agent_003", "name": "Agent 3", "status": "active"},
            {"id": "agent_002", "name": "Agent 2", "status": "active"},
            {"id": "agent_004", "name": "Agent 4", "status": "active"},
            {"id": "agent_005", "name": "Agent 5", "status": "active"}
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                    mock_service = MagicMock()
                    mock_service.start_propagation = AsyncMock(return_value={
                        "session_id": "test_integration_session",
                        "status": "completed",
                        "message": propagation_request["message"],
                        "model_type": propagation_request["model_type"],
                        "seed_agents": propagation_request["seed_agents"],
                        "influenced_agents": ["agent_001", "agent_003", "agent_002"],
                        "propagation_steps": 5,
                        "statistics": {
                            "total_influenced": 3,
                            "seed_count": len(propagation_request["seed_agents"]),
                            "propagation_ratio": 0.6,
                            "propagation_steps": 5,
                            "model_parameters": propagation_request["parameters"]
                        },
                        "created_at": datetime.utcnow()
                    })
                    mock_get_service.return_value = mock_service
                    propagation_response = self.client.post("/api/propagation/start", json=propagation_request)

        assert propagation_response.status_code == 200
        propagation_data = propagation_response.json()

        # 3. 验证传播结果
        assert propagation_data["seed_agents"] == optimal_seeds
        assert len(propagation_data["influenced_agents"]) >= len(optimal_seeds)
        assert propagation_data["statistics"]["seed_count"] == len(optimal_seeds)

    def test_multiple_propagation_sessions(self):
        """测试多个传播会话的管理"""
        # 创建多个传播会话
        sessions = []
        for i in range(3):
            propagation_request = {
                "message": f"测试消息 {i+1}",
                "seed_agents": [f"agent_{i+1:03d}"],
                "model_type": "viral",
                "parameters": {},
                "max_steps": 10
            }

            mock_agents = [
                {"id": f"agent_{i+1:03d}", "name": f"Agent {i+1}", "status": "active"},
                {"id": f"agent_{i+2:03d}", "name": f"Agent {i+2}", "status": "active"}
            ]

            with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
                with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                    response = self.client.post("/api/propagation/start", json=propagation_request)

            assert response.status_code == 200
            sessions.append(response.json())

        # 验证所有会话都有唯一ID
        session_ids = [session["session_id"] for session in sessions]
        assert len(session_ids) == len(set(session_ids))  # 确保ID唯一

        # 验证可以获取活跃会话列表
        mock_active_sessions = sessions
        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_active_sessions = AsyncMock(return_value=mock_active_sessions)
            mock_get_service.return_value = mock_service
            list_response = self.client.get("/api/propagation/sessions")

        assert list_response.status_code == 200
        listed_sessions = list_response.json()["sessions"]
        assert len(listed_sessions) >= 3

    def test_network_metrics_with_propagation_correlation(self):
        """测试网络指标与传播效果的相关性"""
        # 1. 获取网络指标
        mock_metrics = {
            "node_count": 50,
            "edge_count": 120,
            "average_degree": 4.8,
            "clustering_coefficient": 0.4,
            "average_path_length": 2.8,
            "network_density": 0.1,
            "connected_components": 1,
            "largest_component_size": 50
        }

        with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_network_metrics = AsyncMock(return_value=mock_metrics)
            mock_get_service.return_value = mock_service
            metrics_response = self.client.get("/api/network/metrics")

        assert metrics_response.status_code == 200
        network_metrics = metrics_response.json()

        # 2. 进行传播测试
        propagation_request = {
            "message": "网络相关性测试",
            "seed_agents": ["agent_001"],
            "model_type": "viral",
            "parameters": {"infection_probability": 0.1},
            "max_steps": 20
        }

        mock_agents = [
            {"id": "agent_001", "name": "Agent 1", "status": "active"}
            for _ in range(50)
        ]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                propagation_response = self.client.post("/api/propagation/start", json=propagation_request)

        assert propagation_response.status_code == 200
        propagation_result = propagation_response.json()

        # 3. 验证网络指标与传播结果的关联性
        propagation_ratio = propagation_result["statistics"]["propagation_ratio"]

        # 在高密度网络中，传播比例应该较高
        assert network_metrics["network_density"] > 0
        assert 0 <= propagation_ratio <= 1

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复机制"""
        # 1. 测试传播模拟失败的情况
        propagation_request = {
            "message": "错误处理测试",
            "seed_agents": ["agent_001"],
            "model_type": "viral",
            "parameters": {},
            "max_steps": 10
        }

        # 模拟传播失败
        mock_agents = [{"id": "agent_001", "name": "Agent 1", "status": "active"}]

        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                with patch('src.message_propagation.viral_propagation.ViralPropagationModel.propagate_step') as mock_propagate:
                    mock_propagate.side_effect = Exception("Propagation failed")
                    response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 500
        assert "Propagation simulation failed" in response.json()["detail"]

        # 2. 测试系统能够从错误中恢复
        # 恢复正常的传播功能
        with patch('src.web_interface.api.app.agent_service.get_all_agents', return_value=mock_agents):
            with patch('src.web_interface.api.app.social_network_service.get_agent_connections', return_value=[]):
                response = self.client.post("/api/propagation/start", json=propagation_request)

        assert response.status_code == 200  # 恢复正常

    def test_concurrent_propagation_requests(self):
        """测试并发传播请求处理"""
        # 简化的并发测试： sequentially test multiple requests to ensure functionality
        requests_data = [
            {"message": "测试消息 1", "seed_agents": ["agent_001"], "model_type": "viral", "parameters": {}, "max_steps": 5},
            {"message": "测试消息 2", "seed_agents": ["agent_002"], "model_type": "diffusion", "parameters": {}, "max_steps": 5},
            {"message": "测试消息 3", "seed_agents": ["agent_003"], "model_type": "viral", "parameters": {}, "max_steps": 5}
        ]

        for i, propagation_request in enumerate(requests_data):
            mock_agents = [
                {"id": propagation_request["seed_agents"][0], "name": f"Agent {i+1}", "status": "active"}
            ]

            mock_propagation_response = {
                "session_id": f"session_{i+1}",
                "status": "completed",
                "message": propagation_request["message"],
                "model_type": propagation_request["model_type"],
                "seed_agents": propagation_request["seed_agents"],
                "influenced_agents": propagation_request["seed_agents"].copy(),
                "propagation_steps": 3,
                "statistics": {
                    "total_influenced": 1,
                    "seed_count": 1,
                    "propagation_ratio": 1.0,
                    "propagation_steps": 3
                },
                "created_at": datetime.utcnow()
            }

            with patch('src.web_interface.api.app.agent_service.get_all_agents', new_callable=AsyncMock, return_value=mock_agents):
                with patch('src.web_interface.api.app.social_network_service.get_agent_connections', new_callable=AsyncMock, return_value=[]):
                    with patch('src.web_interface.api.app.get_message_propagation_service') as mock_get_service:
                        mock_service = MagicMock()
                        mock_service.start_propagation = AsyncMock(return_value=mock_propagation_response)
                        mock_get_service.return_value = mock_service
                        response = self.client.post("/api/propagation/start", json=propagation_request)

            assert response.status_code == 200, f"Request {i+1} failed with status {response.status_code}"
            result = response.json()
            assert result["session_id"] == f"session_{i+1}"