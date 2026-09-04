import os
import sys
from loguru import logger

# # 这行代码配置了日志记录的基本设置。它调用 logging.basicConfig()，这会对日志记录进行基本配置，例如设置日志记录格式、日志文件等。
# # 这里没有提供具体参数，所以使用默认配置，这通常包括在控制台输出日志消息。
# logging.basicConfig()
#
# # 这行代码获取名为 "langchain.retrievers.multi_query" 的日志记录器，并将其日志级别设置为 INFO。
# # 这样，任何由这个记录器产生的 INFO 级别及以上的日志消息（INFO、WARNING、ERROR、CRITICAL）都会被输出。
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
#
# # 这行代码获取名为 "langchain.retrievers.re_phraser" 的日志记录器，并将其日志级别设置为 INFO。
# # 同样，任何由这个记录器产生的 INFO 级别及以上的日志消息都会被输出。
# logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

def setup_logger(log_dir="logs"):
    """
    配置日志
    Args:
        log_dir: 日志目录
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 移除默认处理器
    logger.remove()
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    # 添加文件处理器
    logger.add(
        os.path.join(log_dir, "doc_qa_{time:YYYY-MM-DD}.log"),
        rotation="00:00",  # 每天轮换
        retention="30 days",  # 保留30天
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    return logger