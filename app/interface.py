import os
from configparser import ConfigParser
from agent_data import AgentData

from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

import gymnasium as gym


class EnvironmentDisplayFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(EnvironmentDisplayFrame, self).__init__(master, *args, **kwargs)
        self.padding = int(str(self.cget("padding")[0]))
        print(type(self.padding))
        print(self.padding)

        self.is_playing = False

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.image = PhotoImage(file="breakout.gif")
        self.render_window = ttk.Label(self, image=self.image)

        self.render_window.grid(column=0, row=0, sticky="nwes")

        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array", autoreset=True, full_action_space=False,
                            frameskip=1)

        self.max_size = (self.env.observation_space.shape[0] * 4, self.env.observation_space.shape[1] * 4)

        self.env.reset()

    def toggle_playing(self, playing=True):
        self.is_playing = playing
        self.render_env()

    def render_env(self):
        if self.is_playing:
            self.env.step(self.env.action_space.sample())

            env_img = self.env.render()
            pillow_img = Image.fromarray(env_img)

            self.update()  # Update widget to get the real width and height values
            # Take padding into account, otherwise, the widget will continue growing (because width and height return
            # the size + padding)
            pillow_img = pillow_img.resize((
                min(self.render_window.winfo_width() - self.padding, self.max_size[1]),
                min(self.render_window.winfo_height() - self.padding, self.max_size[0])
            ))

            self.image = ImageTk.PhotoImage(pillow_img)

            self.render_window["image"] = self.image

            self.after(1, self.render_env)


class HeaderFrame(ttk.Frame):
    def __init__(self, master: Misc, env_display: EnvironmentDisplayFrame, *args, **kwargs):
        super(HeaderFrame, self).__init__(master, *args, **kwargs)

        self.env_display = env_display

        test_agent_btn = Button(self, text="Test Agent", command=self.__test_agent_callback)
        test_agent_btn.grid(column=0, row=0)

    def __test_agent_callback(self):
        self.env_display.toggle_playing(playing=not self.env_display.is_playing)


class EnvironmentListFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(EnvironmentListFrame, self).__init__(master, *args, **kwargs)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        env_choices = ["ALE/Pong-v5", "ALE/Breakout-v4"]
        env_choices_var = StringVar(value=env_choices)

        env_list_box = Listbox(self, width=40, height=30, listvariable=env_choices_var, background="green")
        env_list_box.grid(column=0, row=0, sticky=(N, E, S, W))


class AgentListFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(AgentListFrame, self).__init__(master, *args, **kwargs)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        dirs_name = os.listdir("agents/")

        agents: [AgentData] = []
        for dir_name in dirs_name:
            agents.append(AgentData(config_path=os.path.join("agents", dir_name, "config.ini")))

        agent_choices = [name.display_name for name in agents]

        agent_choices_var = StringVar(value=agent_choices)
        agent_list_box = Listbox(self, width=60, height=30, listvariable=agent_choices_var, background="green")
        agent_list_box.grid(column=0, row=0, sticky=(N, E, S, W))
