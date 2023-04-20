from typing import Callable, List, Optional
from app.utils.typing import EnvConfig, AgentConfig

import os
import json

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from app.widgets.env_display import EnvironmentDisplayFrame

import webbrowser


class HeaderFrame(ttk.Frame):
    def __init__(self, master: Misc, env_display: EnvironmentDisplayFrame, *args, **kwargs):
        super(HeaderFrame, self).__init__(master, *args, **kwargs)

        self.env_display = env_display

        test_agent_btn = ttk.Button(self, text="Play/Pause", command=self.__test_agent_callback)

        test_agent_btn.grid(column=0, row=0)

    def __test_agent_callback(self):
        self.env_display.toggle_playing(playing=not self.env_display.is_playing)


class EnvironmentListFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, on_select_item: Callable[[EnvConfig], None], *args, **kwargs):
        super(EnvironmentListFrame, self).__init__(master, *args, **kwargs)

        self.select_item_callback = on_select_item

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.env_names: [str] = []
        self.configs: List[EnvConfig] = []

        self.envs_dir_abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "envs")
        self.env_dirs = os.listdir(self.envs_dir_abs_path)
        for env_dir in self.env_dirs:
            with open(os.path.join(self.envs_dir_abs_path, env_dir, "config.json")) as config_file:
                config_dict = json.load(config_file)
                config_dict["path"] = os.path.join(self.envs_dir_abs_path, env_dir)
                self.env_names.append(config_dict["env_name"])
                self.configs.append(config_dict)

        # noinspection PyTypeChecker
        env_choices_var = StringVar(value=self.env_names)

        self.env_list_box = Listbox(self, width=40, height=30, listvariable=env_choices_var)
        self.env_list_box.grid(column=0, row=0, sticky=N + E + S + W)
        self.env_list_box.bind("<<ListboxSelect>>", self.__event_listbox_selected)

    def __event_listbox_selected(self, e):
        if len(self.env_list_box.curselection()) > 0:
            self.select_item_callback(self.configs[self.env_list_box.curselection()[0]])


class AgentListFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, on_select_item: Callable[[EnvConfig, AgentConfig], None], *args, **kwargs):
        """

        :param master: parent of the frame
        :param on_select_item: func(env_name, agent_name, path_to_agent)
        :param args:
        :param kwargs:
        """
        super(AgentListFrame, self).__init__(master, *args, **kwargs)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.select_item_callback = on_select_item

        self.env_config: Optional[EnvConfig] = None

        self.agent_names = []
        # noinspection PyTypeChecker
        self.agent_choices_var = StringVar(value=self.agent_names)

        self.agent_list_box = Listbox(self, width=40, height=30, listvariable=self.agent_choices_var)
        self.agent_list_box.grid(column=0, row=0, sticky=N + E + S + W)
        self.agent_list_box.bind("<<ListboxSelect>>", self.__event_listbox_select)

    def __event_listbox_select(self, e):
        if self.env_config is not None and len(self.agent_list_box.curselection()) > 0:
            selected_agent = self.env_config["agents"][self.agent_list_box.curselection()[0]]
            self.select_item_callback(self.env_config, selected_agent)

    def update_list(self, env_config: EnvConfig):
        """
        Put in the agents compatible with the selected environment
        :param env_config:
        """
        self.env_config = env_config

        self.agent_names.clear()
        for agent in self.env_config["agents"]:
            self.agent_names.append(agent["display_name"])

        # noinspection PyTypeChecker
        self.agent_choices_var.set(self.agent_names)


class NetworkVisualisationFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(NetworkVisualisationFrame, self).__init__(master, *args, **kwargs)
        self.padding = int(str(self.cget("padding")[0]))

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.image = None  # Keep reference to the image during app lifetime. Required by the Label widget
        self.image_label = ttk.Label(self, image=self.image, anchor="center")

        self.image_label.grid(column=0, row=0, sticky=N + S + E + W)

    def update_network_image(self, network_type: str):
        if network_type == "Conv":
            image = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "images",
                                            "cnn_graph.png"))

            new_width = self.image_label.winfo_width() - self.padding
            new_height = new_width * image.size[1] // image.size[0]
            resized_image = image.resize((new_width, new_height))

            self.image = ImageTk.PhotoImage(resized_image)
            self.image_label["image"] = self.image
        elif network_type == "Random":
            self.image = None
            self.image_label["image"] = self.image
        else:
            image = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "images",
                                            "swin_transformer_graph.png"))

            new_width = self.image_label.winfo_width() - self.padding
            new_height = new_width * image.size[1] // image.size[0]
            resized_image = image.resize((new_width, new_height))

            self.image = ImageTk.PhotoImage(resized_image)

            self.image_label["image"] = self.image
