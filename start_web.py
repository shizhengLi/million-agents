#!/usr/bin/env python3
"""
简单的Web界面启动脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import uvicorn
    from web_interface.api.app import create_app

    print("🚀 启动Million Agents Web Interface...")
    print("📱 访问地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    print("⚠️  按Ctrl+C停止服务器")

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请安装依赖: pip install fastapi uvicorn")

    # 启动简单HTTP服务器作为备选
    import http.server
    import socketserver
    import webbrowser

    PORT = 8080
    os.chdir(Path(__file__).parent / "src" / "web_interface")

    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)

    print(f"🔄 启动简单HTTP服务器，端口: {PORT}")
    print(f"📱 访问地址: http://localhost:{PORT}")
    print("⚠️  注意：这是静态演示版本")

    webbrowser.open(f"http://localhost:{PORT}")
    httpd.serve_forever()

except KeyboardInterrupt:
    print("\n👋 服务器已停止")
except Exception as e:
    print(f"❌ 启动失败: {e}")