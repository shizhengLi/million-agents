#!/usr/bin/env python3
"""
Million Agents Web Interface Launcher
Web管理界面启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_dependencies():
    """检查依赖是否安装"""
    required_modules = ['fastapi', 'uvicorn']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"❌ 缺少依赖模块: {', '.join(missing_modules)}")
        print("请运行以下命令安装依赖:")
        print("pip install fastapi uvicorn jinja2 python-multipart httpx websockets")
        return False

    print("✅ 所有依赖已安装")
    return True


def create_simple_server():
    """创建简单的HTTP服务器用于静态文件"""
    import http.server
    import socketserver
    import threading

    # 切换到项目根目录
    os.chdir(Path(__file__).parent)

    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="src/web_interface", **kwargs)

        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()

    PORT = 8080
    Handler = CustomHTTPRequestHandler

    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        print(f"🚀 启动简单HTTP服务器，端口: {PORT}")
        print(f"📱 访问地址: http://localhost:{PORT}")
        print("📄 静态文件服务器已启动（仅用于演示）")
        print("⚠️  注意：这是一个静态演示版本，API调用将返回模拟数据")

        # 自动打开浏览器
        webbrowser.open(f"http://localhost:{PORT}")

        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")


def create_fastapi_server():
    """创建FastAPI服务器"""
    try:
        # 添加src目录到Python路径
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from web_interface.api.app import create_app
        import uvicorn

        app = create_app()

        print("🚀 启动FastAPI服务器...")
        print("📱 访问地址: http://localhost:8000")
        print("📚 API文档: http://localhost:8000/docs")
        print("⚠️  按Ctrl+C停止服务器")

        # 自动打开浏览器
        webbrowser.open("http://localhost:8000")

        # 启动服务器（不使用reload避免警告）
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except ImportError as e:
        print(f"❌ 无法导入FastAPI应用: {e}")
        print("🔄 切换到简单HTTP服务器模式...")
        create_simple_server()
    except Exception as e:
        print(f"❌ 启动FastAPI服务器失败: {e}")
        print("🔄 切换到简单HTTP服务器模式...")
        create_simple_server()


def show_usage():
    """显示使用说明"""
    print("""
🌟 Million Agents Web Interface Launcher
=====================================

用法:
    python run_web_interface.py [选项]

选项:
    --simple    使用简单HTTP服务器（仅静态文件）
    --fastapi   强制使用FastAPI服务器
    --help      显示此帮助信息

功能:
    📊 仪表板 - 系统概览和核心指标
    🤖 智能体管理 - 智能体CRUD操作和管理
    🕸️  网络可视化 - 社交网络关系可视化
    📈 系统指标 - 详细的性能和统计数据

特性:
    ✅ 响应式设计，支持移动端
    ✅ 实时数据更新（WebSocket）
    ✅ 模拟数据支持（演示用）
    ✅ 完整的测试覆盖（29个测试用例）
    ✅ 现代化UI设计

注意:
    • 如果FastAPI未安装，将自动切换到简单模式
    • 简单模式仅提供静态文件演示
    • 生产环境请使用FastAPI模式
    • API数据目前使用模拟数据
""")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            show_usage()
            return
        elif sys.argv[1] == '--simple':
            print("🔄 强制使用简单HTTP服务器模式...")
            create_simple_server()
            return
        elif sys.argv[1] == '--fastapi':
            print("🔄 强制使用FastAPI服务器模式...")
            if not check_dependencies():
                return
            create_fastapi_server()
            return

    # 自动模式：优先尝试FastAPI，失败则使用简单模式
    print("🔍 检查运行环境...")

    if check_dependencies():
        print("✅ 使用FastAPI服务器模式")
        create_fastapi_server()
    else:
        print("⚠️  依赖检查失败，使用简单HTTP服务器模式")
        create_simple_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"❌ 程序异常退出: {e}")
        sys.exit(1)