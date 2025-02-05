import logging
import os


class Log(object):
    def __init__(self,module,filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./result.txt/'):
            os.makedirs('./result.txt/')
        handler = logging.FileHandler('./result.txt/'+filename+'.result.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)
