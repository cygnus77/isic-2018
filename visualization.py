from datetime import datetime
import visdom

class Visualization:
    def __init__(self, title, env_name=None):
        if env_name is None:
            env_name = "Lesion %s" % str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.title = title

    def plot_loss(self, loss, step, name):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            name=name,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title=self.title,
            )
        )