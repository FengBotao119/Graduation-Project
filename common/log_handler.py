import os
import logging.handlers
import logging.config
import sys
#待解决问题 如何把输出日志放在合适的位置 已经解决
_logger = None
def get_logger():
    global _logger

    if _logger is not None:  #解决重复调用问题
        return _logger

    #创建一个日志器 级别为DEBUG
    _logger = logging.getLogger("root")
    _logger.setLevel(logging.DEBUG)

    #设置输出格式
    formatter1 = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter2 = '%(asctime)s - %(name)s  - %(message)s'
    #创建第一个处理器 
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.INFO)
    handler_stdout.setFormatter(logging.Formatter(formatter2))

    #创建第二个处理器
    handler_log = logging.FileHandler('train.log')
    handler_log.setLevel(logging.DEBUG)
    handler_log.setFormatter(logging.Formatter(formatter1))
    
    #添加处理器
    _logger.addHandler(handler_stdout)
    _logger.addHandler(handler_log)
    return _logger

if __name__=='__main__':
    logger = get_logger()
    logger.info('i am fbt')