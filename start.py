#!/usr/bin/env python3
"""
启动脚本 - 读取.env配置启动前后端服务
"""

import os
import sys
import subprocess
from pathlib import Path


def load_env():
    """加载.env文件"""
    env_file = Path(__file__).parent / ".env"
    env_vars = {}

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value

    return env_vars


def main():
    env_vars = load_env()

    backend_port = env_vars.get("BACKEND_PORT", "8000")
    frontend_port = env_vars.get("FRONTEND_PORT", "3000")

    print(f"后端端口: {backend_port}")
    print(f"前端端口: {frontend_port}")
    print()

    # 启动后端
    print("启动后端服务...")
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.web_interface.api.app:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            backend_port,
            "--reload",
        ],
        cwd=Path(__file__).parent,
    )

    # 启动前端
    print("启动前端服务...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=Path(__file__).parent / "frontend",
        env={**os.environ, "BACKEND_PORT": backend_port},
    )

    print(f"\n服务启动完成!")
    print(f"前端: http://localhost:{frontend_port}")
    print(f"后端: http://localhost:{backend_port}")
    print("\n按 Ctrl+C 停止所有服务")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.terminate()


if __name__ == "__main__":
    main()
