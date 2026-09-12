# syntax=docker/dockerfile:1

# 使用 slim 变体：完整版 python 镜像自带大量编译工具链，体积可超 1GB；
# slim 版约 130MB，运行本项目所需的运行时依赖完全够用。
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE：容器内无需 .pyc 缓存，避免产生无用文件
# PYTHONUNBUFFERED：日志不缓冲、直接写 stdout，docker logs 才能实时看到输出
# PIP_NO_CACHE_DIR：不留 pip 下载缓存，减小镜像层体积
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 关键：先只拷贝依赖清单并安装，再拷贝业务代码。
# Docker 分层缓存的 key 是「指令 + 输入文件内容」，因此改代码不会触发依赖重装。
# 依赖安装通常是最慢的一层，缓存命中能把重建时间从几分钟降到几秒。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 如需启用 ChromaDB 向量库，取消下面这行注释（会额外增加约 200MB 镜像体积）：
# RUN pip install --no-cache-dir "chromadb>=0.5.0"

# --chown 直接把文件属主设为运行用户，避免后续再跑一次 chown 增加镜像层
RUN useradd --create-home --shell /usr/sbin/nologin appuser
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval/ ./eval/
COPY --chown=appuser:appuser pytest.ini ./

# 以非 root 用户运行，缩小容器被攻破时的权限范围（也是安全基线扫描的常规要求）
USER appuser

EXPOSE 8000

# 健康检查直接打 /health。
# 不装 curl：slim 镜像默认没有 curl，仅为健康检查引入一个包并不划算；
# 用 Python 标准库 urllib 即可，零额外依赖。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).getcode()==200 else 1)"

# 必须监听 0.0.0.0 而不是 127.0.0.1：
# 容器内的 127.0.0.1 只对容器自身可见，宿主机与负载均衡都无法连入。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
