import datetime

import pandas as pd
import pareto as pareto
import numpy as np


def find_pareto_frontiers(data, input_axes, n_front=10):
    data.reset_index(drop=True, inplace=True)
    data.loc[:, "index"] = np.array(data.index)
    pareto_axe = np.ones(data.shape[0]) * -1
    i = 0
    while -1 in pareto_axe:
        # Select data to be studied
        selected_data = data.loc[np.where(pareto_axe == -1)[0]]

        # define the objective function column indices
        of_cols = [selected_data.axes[1].to_list().index(column) for column in input_axes]

        # sort
        nondominated = pareto.eps_sort([list(selected_data.itertuples(False))], of_cols)

        # convert multi-dimension array to DataFrame
        pareto_axe[
            pd.DataFrame.from_records(
                nondominated,
                columns=list(selected_data.columns.values)
            )["index"]
        ] = i
        i += 1
        if i >= n_front:
            break
    return pareto_axe


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
