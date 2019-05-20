import tensorflow as tf
import os
import logging

class MyLogger():
    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_writer = tf.summary.FileWriter(log_dir)
        self.txt_logger = logging.getLogger('info')
        self.handler = logging.FileHandler(log_dir+"/log.txt")
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.txt_logger.addHandler(self.handler)
        self.cache_count = 0

    def write_summary_scalar(self, iteration, tag, value):
        # 添加自定义的数据进去
        # iteration代表横轴坐标
        self.log_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)

    def add_sess_graph(self, graph):
        self.log_writer.add_graph(graph)

    def add_info_txt(self, message):
        self.txt_logger.info(message)

    def add_warning_txt(self, message):
        self.txt_logger.warning(message)