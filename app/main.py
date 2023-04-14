from tkinter import *
from tkinter import ttk

from interface import HeaderFrame, AgentListFrame, EnvironmentListFrame, EnvironmentDisplayFrame

class MainApplication(ttk.Frame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(MainApplication, self).__init__(master, *args, **kwargs)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        env_display = EnvironmentDisplayFrame(self, text="Environment Display", padding=10)
        env_display.grid(column=1, row=1, sticky=(N, S, E, W))

        header = HeaderFrame(self, env_display=env_display, style="BGMagenta.TFrame")
        header.grid(column=0, row=0, sticky=(N, S, E, W))

        environment_list = EnvironmentListFrame(self, text="Environment List", padding=15, style="BGPink.TFrame")
        environment_list.grid(column=0, row=1, sticky=(N, S, E, W))

        agent_list = AgentListFrame(self, text="Agent List", padding=15, style="BGMagenta.TFrame")
        agent_list.grid(column=0, row=2, sticky=(N, S, E, W))







if __name__ == '__main__':
    root = Tk()
    root.title("FYP Deep Reinforcement Learning Comparison")
    #root.geometry("1920x1080")

    s = ttk.Style()
    s.configure("BGRed.TFrame", background="red")
    s.configure("BGPink.TFrame", background="pink")
    s.configure("BGMagenta.TFrame", background="magenta")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    MainApplication(root, style="BGRed.TFrame").grid(column=0, row=0, sticky=(N, E, S, W))



    root.mainloop()
