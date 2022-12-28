#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: start_training.py
Authors: Christian Bergler
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import re
import sys
import logging

class setup_training(object):
    
    def __init__(self, config, log_level="debug"):
        self.logger = None
        self.config_path = config
        self.config_data = dict()
        self.log_level = log_level
        self.loglevels = ["debug", "error", "warning", "info"]

    def init_logger(self):
        self.logger = logging.getLogger('training animal-spot')
        stream_handler = logging.StreamHandler()
        if self.log_level.lower() == self.loglevels[0]:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        elif self.log_level.lower() == self.loglevels[1]:
            self.logger.setLevel(logging.ERROR)
            stream_handler.setLevel(logging.ERROR)
        elif self.log_level.lower() == self.loglevels[2]:
            self.logger.setLevel(logging.WARNING)
            stream_handler.setLevel(logging.WARNING)
        elif self.log_level.lower() == self.loglevels[3]:
            self.logger.setLevel(logging.INFO)
            stream_handler.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    
    def read_config(self):
        config_file = open(self.config_path, 'r')
        for element in config_file:
            if element[0] != "#" and element[0] != " " and element[0] != "" and element[0] != "\n":
                key = element.strip().split("=")[0]
                value = element.strip().split("=")[1]
                if value == "true":
                    self.config_data[key] = ""
                elif value == "false":
                    continue
                else:
                    self.config_data[key] = value
        self.logger.info("Config Data: " + str(self.config_data))
                
    def start_training(self):
        train_cmd = "python -W ignore::UserWarning "+self.config_data["src_dir"]+"/main.py"
        for key in self.config_data:
            if key != "src_dir":
                train_cmd = train_cmd + " " + "--" + key + " " + self.config_data.get(key)
        train_cmd = re.sub(r'\s+', " ", train_cmd).strip()
        self.logger.info("Training Command: " + train_cmd)
        self.logger.info("Start Training!!!")
        os.system(train_cmd)
        

if __name__ == '__main__':
    if len(sys.argv) == 2:
        trainer = setup_training(config=sys.argv[1])
        trainer.init_logger()
        trainer.read_config()
        trainer.start_training()
    else:
        raise Exception('Invalid Number of Cmd Parameter. Only one argument -> path of config file')
