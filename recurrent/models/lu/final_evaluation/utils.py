import logging

# create logger for individual name and individual cross_validation folder
def setup_logger(logger_name, log_file, level=logging.DEBUG):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    if (l.hasHandlers()):
        l.handlers.clear()
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)