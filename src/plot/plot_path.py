import matplotlib.pylab as plt
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
import seaborn as sb


class Plotting:

    @staticmethod
    def plot_new_fig(num):
        sb.set()
        sb.set_theme(context='paper', style='whitegrid')
        # sb.set(font_scale=3)

        plt.rc('font', family='serif', serif='Times')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('axes', labelsize=18)
        plt.rc('legend', fontsize=24)

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
        return fig, ax

    @staticmethod
    def plot_set_theme(theme):
        sb.set_theme(context='notebook', style=theme)

    @staticmethod
    def plot_path_orientation(path=[], color='b', style='-', linewidth=2, label=None, master_path=False):
        '''
        plots a sequence of vehicle states of as a path
        '''

        from matplotlib.patches import FancyArrowPatch
        axis = plt.gca()
        if (path == []):
            print("No path provided!")
            return

        x = []
        y = []

        if master_path:
            start_frame = 0
        else:
            start_frame = 0
        for _i in range(start_frame, len(path)):
            if (type(path[_i]) == VehicleState):
                position = path[_i].get_position()
                orientation = path[_i].get_rotation()
                d = Vector3(vec=[0, 0, 0.1])
                d = orientation.rotate(d)
                dx = d[0]
                dy = d[2]
                arrow = FancyArrowPatch(
                    (position[0], position[2]),
                    (position[0] + 2*dx, position[2] + 2*dy),
                    arrowstyle='-|>',
                    shrinkA=1/2.0,
                    shrinkB=1/2.0,
                    linewidth=1,
                    mutation_scale=10,
                    zorder=0,
                    linestyle='-',
                    alpha=.8
                )
                axis.add_patch(arrow)
            else:
                position = path[_i]
            x.append(position[0])
            y.append(position[2])
        if label == None:
            plt.plot(x, y, '-*', color=color, linewidth=linewidth)
        else:
            plt.plot(x, y, color=color, linestyle=style, linewidth=linewidth, label=label)
            plt.legend()

    @staticmethod
    def plot_path(path=[], color='b', style='-', linewidth=2, label=None):
        '''
        plots a sequence of vehicle states of as a path
        '''

        if (path == []):
            print("No path provided!")
            return

        x = []
        y = []
        for _i in range(len(path)):
            if (type(path[_i]) == VehicleState):
                position = path[_i].get_position()
            else:
                position = path[_i]
            x.append(position[0])
            y.append(position[2])
        if label == None:
            plt.plot(x, y, color=color, linestyle=style, linewidth=linewidth)
        else:
            plt.plot(x, y, color=color, linestyle=style, linewidth=linewidth, label=label)
            plt.legend()

    @staticmethod
    def plot_prediction(prediction, color="b", style="*", markersize=2):
        '''
        plots the point
        '''
        if (type(prediction) == VehicleState):
            position = prediction.get_position()
        else:
            position = prediction

        plt.plot(position[0], position[2], style, color=color, markersize=markersize)

    @staticmethod
    def plot_predictions(predictions, color="g", style="*", markersize=2):
        '''
        plots a sequence of points
        '''

        if (type(predictions[0]) == VehicleState):
            position = predictions[0].get_position()
        else:
            position = predictions[0]

        for _i in range(len(predictions)):
            position = predictions[_i].get_position()
            plt.plot(position[0], position[2], style, color=color, markersize=markersize)

    @staticmethod
    def plot_sequence(predictions=[], color="b", style="*", markersize=2, label=None):
        '''
        plots the predicted points
        '''
        for _i in range(len(predictions)-1):
            if (type(predictions[_i]) == VehicleState):
                position = predictions[_i].get_position()
            else:
                position = predictions[_i]
            plt.plot(position[0], position[2], style, color=color, markersize=markersize)

        if (label == None):
            if (type(predictions[len(predictions)-1]) == VehicleState):
                position = predictions[len(predictions)-1].get_position()
            else:
                position = predictions[len(predictions)-1]
            plt.plot(position[0], position[2], style, color=color, markersize=markersize)

        else:
            if (type(predictions[len(predictions)-1]) == VehicleState):
                position = predictions[len(predictions)-1].get_position()
            else:
                position = predictions[len(predictions)-1]

            plt.plot(position[0], position[2], style, color=color, markersize=markersize, label=label)
            plt.legend()

    @staticmethod
    def plot_error(clock=[], error_data=[], color="b", style="*", markersize=2, label=None):
        '''
        plots the error data
        '''
        if label is None:
            plt.plot(clock, error_data, style, color=color, markersize=markersize)
        else:
            plt.plot(clock, error_data, style, color=color, markersize=markersize, label=label)
            plt.legend()

    def show(self, equal=False, xlim=None, ylim=None):
        if(equal):
            plt.axis('equal')
            
        if (xlim!=None):
            plt.xlim(xlim)
        if (ylim!=None):
            plt.ylim(ylim)
        plt.show(block=True)
