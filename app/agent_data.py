import os
from configparser import ConfigParser
from tkinter import *


class AgentData:
    def __init__(self, config_path: str = None, display_name: str = ""):
        if config_path is None:
            self.display_name = display_name
        else:
            self.path = config_path
            self.load_from_file(config_path)

    def load_from_file(self, path):
        self.path = path
        config_parser = ConfigParser()
        config_parser.read(path)

        self.display_name = config_parser["metadata"].get("display_name")
