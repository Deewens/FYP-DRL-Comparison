from tkinter import *
from tkinter import ttk
from tkinter import font

from app.widgets.interface import HeaderFrame, AgentListFrame, EnvironmentListFrame, NetworkVisualisationFrame
from app.widgets.env_display import EnvironmentDisplayFrame
from app.utils.typing import EnvConfig, AgentConfig
import sv_ttk



class MainApplication(ttk.Frame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(MainApplication, self).__init__(master, *args, **kwargs)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        self.env_display = EnvironmentDisplayFrame(self, text="Environment Display", padding=10)
        header = HeaderFrame(self, env_display=self.env_display)
        agent_list = AgentListFrame(self, text="Agent List", padding=15, on_select_item=self.__on_select_agent)

        environment_list = EnvironmentListFrame(self, text="Environment List", padding=15, on_select_item=agent_list.update_list)
        self.network_viz = NetworkVisualisationFrame(self, text="Brain", padding=10)


        header.grid(column=0, row=0, sticky=N + S + E + W)
        environment_list.grid(column=0, row=1, sticky=N + S + E + W)
        agent_list.grid(column=1, row=1, sticky=N + S + E + W)
        self.env_display.grid(column=2, row=1, sticky=N + S + E + W)
        self.network_viz.grid(column=2, row=2, sticky=N + S + E + W)

    def __on_select_agent(self, env_config: EnvConfig, agent_config: AgentConfig):
        self.env_display.setup_agent(env_config, agent_config)
        if agent_config["type"] == "Conv":
            self.network_viz.update_network_image("Conv")
        elif agent_config["type"] == "Random":
            self.network_viz.update_network_image("Random")
        else:
            self.network_viz.update_network_image("Swin")



if __name__ == '__main__':
    root = Tk()

    default_font = font.nametofont("TkDefaultFont")
    default_font.config(family="Helvetica", size=42)


    root.title("FYP Deep Reinforcement Learning Comparison")
    # root.geometry("800x400")
    root.geometry("1920x1080")

    sv_ttk.set_theme("light")

    s = ttk.Style()
    s.configure("BGRed.TFrame", background="red")
    s.configure("BGPink.TFrame", background="pink")
    s.configure("BGMagenta.TFrame", background="magenta")
    s.configure('.', font=('Helvetica', 42))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    MainApplication(root).grid(column=0, row=0, sticky=N + S + E + W)

    root.mainloop()
