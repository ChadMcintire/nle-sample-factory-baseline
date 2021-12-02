import matplotlib.pyplot as plt
import numpy as np
import datetime


class GraphBuilder:
    """Builder to store data points and build graphs
    Args:
        graph_types:        Array of Strings, Each string is a key in a dictionary. Each graph_type will store its own
                                array of points

    """

    def __init__(self, graph_types):
        self.graphs_data = {}

        for g in graph_types:
            self.graphs_data[g] = []

    def set_data(self, graph_type, data):
        """
        Set the data for a specific graph_type. This will completely overwrite and previous data stored in that location
        Args:
            graph_type:     String, Type of graph, also the key in graphs_data
            data:           Array of Tuples, Expects an array of tuples for graph_type of 'heat_pos' and 'heat_search'
        """
        self.graphs_data[graph_type] = data

    def append_point(self, graph_type, pt):
        """
        Add the point to the specified graph's data
        Args:
            graph_type:     String, The graph the point should be added to
            pt:             Tuple (x,y) representing a point
        """
        # append point to empty list
        if len(self.graphs_data[graph_type]) == 0:
            self.graphs_data[graph_type].append(pt)
            print("Had no previous point")
            return
        # get last point added
        start = self.graphs_data[graph_type][len(self.graphs_data[graph_type]) - 1]
        all_pts = self._pos_between(end_pt=pt, start_pt=start)
        for i in all_pts:
            self.graphs_data[graph_type].append(i)

    def _pos_between(self, end_pt, start_pt):
        """
        Find all points that lie between two points

        Args:
            end_pt:         Tuple, formatted as (x,y)
            start_pt:       Tuple, formatted (x,y)

        Returns: List of points between start_pt and end_pt. Including end_pt.

        """
        all_pos = []
        dist = None  # number of positions traversed from last_pos to reach pos
        x_d = end_pt[0] - start_pt[0]  # distance from pos.x and last_pos.x
        y_d = end_pt[1] - start_pt[1]  # distance from pos.y and last_pos.y

        diag = True if abs(x_d) == abs(y_d) else False  # if these points lie on a diagonal
        if diag:
            dist = abs(x_d)
        else:
            dist = abs(x_d) if not diag and abs(x_d) > 0 else abs(y_d)

        # pos and last_pos are the same
        if dist == 0:
            return [end_pt]
        # difference between each point on the line, last_pos --> pos
        x_diff = int(x_d / dist) if not diag else 1 if x_d > 0 else -1
        y_diff = int(y_d / dist) if not diag else 1 if y_d > 0 else -1

        # find all positions on line
        for i in range(1, dist):
            i_pos = (start_pt[0] + x_diff * i, start_pt[1] + y_diff * i)
            all_pos.append(i_pos)

        all_pos.append(end_pt)
        return all_pos

    def _prep_data(self, graph_type, transpose=False):
        """
        Prepare all the data for a specific graph to be used for a head map
        Args:
            graph_type:     String, Type of graph
            transpose:      Boolean, True if the prepared data needs to be transposed

        Returns: 2D array of data ready to be loaded into a PyPlot heatmap
        """
        data = self.graphs_data[graph_type]
        # unzip tuples
        x, y = [i[0] for i in data], [i[1] for i in data]
        m = max(x)
        n = max(y)
        out_data = np.zeros([m, n])

        # count number of occurrences for each point
        counts = {}
        for pos in data:
            if pos in counts:
                counts[pos] += 1
            else:
                counts[pos] = 1

        for i in range(len(out_data)):
            for j in range(len(out_data[i])):
                if (i, j) in counts:
                    out_data[i][j] = counts[(i, j)]
        if transpose:
            out_data = np.transpose(out_data)

        return out_data

    def save_graphs(self, loc=""):
        e = datetime.datetime.now()
        time_prefix = "%s-%s-%s_%s_%s_%s" % (e.day, e.month, e.year, e.hour, e.minute, e.second)
        for key in self.graphs_data:
            fname = time_prefix + key
            data = self._prep_data(key, transpose=True)
            path = loc + ("/%s.png" % fname)
            plt.imsave(fname=path, arr=data, cmap='coolwarm')