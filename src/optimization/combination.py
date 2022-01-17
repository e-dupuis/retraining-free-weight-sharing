import os
from ast import literal_eval

import numpy as np
import pandas as pd
from doepy import build
from tqdm import tqdm

from src import log, scoring, model
from src.approx import clustering
from src.log import append_data


def try_all_sampling(input_model: model.Model, k_list, log_file, num_samples=None, ds_scale=1, ):
    # Define explored designs
    sampling_method_list = [
        build.lhs,
        build.space_filling_lhs,
        build.random_k_means,
        build.maximin,
        build.halton,
        build.uniform_random,
        # build.full_fact,
        # build.frac_fact_res, # error: 'design requires too many base-factors.'
        # build.plackett_burman,
        # build.sukharev,
        build.box_behnken,
        # build.central_composite,
    ]

    # Measure r_square of the regression for each
    result = {}
    max_rsquare = None
    best_method = None
    for sampling_method in tqdm(sampling_method_list):
        print(sampling_method.__name__)
        df = search_space_sampling(input_model,
                                   k_list,
                                   log_file,
                                   sampling_method=sampling_method,
                                   num_samples=num_samples,
                                   ds_scale=ds_scale,
                                   )
        r_square, rmse, = evaluate_regression_model(df.accuracy_loss.values, df.inertia_vector.values)
        print(f"{sampling_method.__name__} with {len(df)} samples: R²={r_square:.2f}, RMSE={rmse:.2f}")
        result[sampling_method.__name__] = r_square
        if max_rsquare is None or r_square > max_rsquare:
            max_rsquare = r_square
            best_method = sampling_method

    # return the best r_square
    print(f"Best method: {best_method.__name__} with linear regression r²={max_rsquare} ")
    return search_space_sampling(input_model,
                                 k_list,
                                 log_file,
                                 sampling_method=best_method,
                                 num_samples=num_samples,
                                 ds_scale=ds_scale,
                                 )


def fix_inertia_vect(df, h5_path):
    # Fix inertia vect issue
    def from_str(a):
        return a.replace("array(", "").replace(", dtype=float32)", "")

    if "inertia_vector" in df.columns and True in df["inertia_vector"].str.contains("array"):
        df["inertia_vector"] = df["inertia_vector"].apply(from_str)
        df.to_hdf(h5_path, key="data")
    return df


def search_space_sampling(input_model: model.Model, k_list, log_file, sampling_method=build.uniform_random,
                          num_samples=None,
                          ds_scale=1):
    # Loading already sampled
    h5_path = os.path.join(log_file, f"samples_exploration_{sampling_method.__name__}_{num_samples}.h5")
    if os.path.exists(h5_path):
        print(f"load {sampling_method.__name__} sampling from {h5_path}")
        samples_exploration_df: pd.DataFrame = pd.read_hdf(h5_path, key="data", index_col=0)
        if isinstance(samples_exploration_df.inertia_vector.values[0], str) or "array" in \
                samples_exploration_df.inertia_vector.values[0]:
            samples_exploration_df = fix_inertia_vect(samples_exploration_df, h5_path)
        samples_exploration_df = samples_exploration_df[samples_exploration_df["accuracy_loss"].notnull()]
        if samples_exploration_df.shape[0] >= num_samples:
            return samples_exploration_df
        else:
            print(f"loaded number of samples {samples_exploration_df.shape[0]} < {num_samples}")
    else:
        samples_exploration_df = None

    # Build unsampled exploration space
    exploration_space = {layer.name: [0, len(k_list[layer.name]) - .001] for layer in
                         input_model.get_layers_of_interests()}
    # Samples selection
    import inspect
    if "num_samples" in inspect.signature(sampling_method).parameters:
        samples = sampling_method(exploration_space, num_samples=num_samples)
    else:
        samples = sampling_method(exploration_space)

    assert list(samples.columns) == list(exploration_space.keys())

    # Reorder
    samples = samples.values.astype(int)
    assert samples.shape[-1] == len(input_model.get_layers_of_interests()), \
        f"Failed: {samples.shape[-1]} == {len(input_model.get_layers_of_interests())}"

    # Check range of the sampled design space
    for i in range(samples.shape[-1]):
        assert len(k_list[input_model.get_layers_of_interests()[i].name]) - 1 == max(
            samples[:, i]), f"{input_model.get_layers_of_interests()[i].name},\
            {exploration_space[input_model.get_layers_of_interests()[i].name]}, \
            {len(k_list[input_model.get_layers_of_interests()[i].name])} < {max(samples[:, i])}"

    # Convert to k_vector
    k_vector_list = np.array(
        [[k_list[input_model.get_layers_of_interests()[i].name][j] for i, j in enumerate(id_vect)] for id_vect in
         samples])
    assert (k_vector_list.shape == samples.shape)

    # Samples evaluation
    tic = log.toc()
    print(f"Testing {len(samples)} samples")
    evaluation_df = input_model.score_approximation_list(k_vector_list, ds_scale=ds_scale, verbose=True,
                                                         origin_str=f"{sampling_method.__name__}_sampling",
                                                         h5_path=h5_path)
    print(f"Tested {len(samples)} samples in {log.toc(tic)}s")

    # Aggregating and returning
    if not isinstance(samples_exploration_df, type(None)):
        samples_exploration_df.append(evaluation_df)
    else:
        samples_exploration_df = evaluation_df

    samples_exploration_df.to_hdf(h5_path, key="data")
    return samples_exploration_df


def population_expansion(input_model: model.Model, k_list: pd.DataFrame, layer_df: pd.DataFrame, var_select: str,
                         population_df: pd.DataFrame, log_file: str, ds_scale: float,
                         max_acl=10 / 100, real_evaluation=False, time_bounded=False, sampling=None):
    from pymoo.optimize import minimize
    from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
    from pymoo.core.problem import Problem
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    from sklearn.linear_model import LinearRegression

    h5_path = os.path.join(log_file, f"population_expansion_{'real' if real_evaluation else 'regression'}.h5")

    if os.path.exists(h5_path):
        print(f"Load expansion exploration from {h5_path}")
        population_df = pd.read_hdf(h5_path, key="data")
        if isinstance(population_df.k_vector, np.ndarray):
            population_df.k_vector = [k_vector.tolist() for k_vector in population_df.k_vector]
        return population_df

    layer_list = [layer.name for layer in input_model.get_layers_of_interests()]

    # Constant params
    W = np.array([input_model.get_layer_weights(layer, force_numpy=True).size for layer in
                  input_model.get_layers_of_interests()])
    N_layer = len(W)
    B = 32

    # Helpers methods
    def extract_k(id_vect):
        result = []
        for i, id in enumerate(id_vect):
            if i >= len(layer_list):
                raise ValueError(f"{i} is not a valid indice for layer list with length {len(layer_list)}")
            if id >= len(k_list[layer_list[i]]):
                raise ValueError(
                    f"{id} is not a valid indice for selected k list of layer {i, layer_list[i]} with length {len(k_list[layer_list[i]])}")
            k = k_list[layer_list[i]][id]
            result.append(k)
        return np.array(result)

    def extract_inertia(id_vect):
        result = []
        for i, id in enumerate(id_vect):
            if i >= len(layer_list):
                raise ValueError(f"{i} is not a valid indice for layer list with length {len(layer_list)}")
            if id >= len(k_list[layer_list[i]]):
                raise ValueError(
                    f"{id} is not a valid indice for selected k list of layer {i, layer_list[i]} with length {len(k_list[layer_list[i]])}")
            inertia_value = layer_df[layer_df.layer == layer_list[i]].set_index("k").loc[
                k_list[layer_list[i]][id]].inertia
            if isinstance(inertia_value, pd.Series):
                inertia_value = inertia_value.values[0]
            result.append(inertia_value)
        return np.array(result)

    def extract_acl(id_vect):
        result = []
        assert len(id_vect) == len(layer_list)
        for i, id in enumerate(id_vect):
            if i >= len(layer_list):
                raise ValueError(f"{i} is not a valid indice for layer list with length {len(layer_list)}")
            if id >= len(k_list[layer_list[i]]):
                raise ValueError(
                    f"{id} is not a valid indice for selected k list of layer {i, layer_list[i]} with length {len(k_list[layer_list[i]])}")
            accuracy_loss_value = layer_df[layer_df.layer == layer_list[i]].set_index("k").loc[
                k_list[layer_list[i]][id]].accuracy_loss
            if isinstance(accuracy_loss_value, pd.Series):
                accuracy_loss_value = accuracy_loss_value.values[0]
            result.append(accuracy_loss_value)
        return np.array(result)

    # CR
    def calc_compression_rate_population(x, W, B):
        return np.array([CR(extract_k(K_list), W, B) for K_list in x[:, ]])

    def CR(K, W, B):
        # multiply compression rate by -1 because optimization algo minimize the value
        return - 1 * np.average(W * B / (W * np.ceil(np.log2(K)) + K * B), weights=W)

    if not real_evaluation:
        ## Model training
        score = 0
        ratio = 1
        while score < 0.8 and ratio * max_acl > 0:
            selected_data = population_df[population_df.accuracy > ratio * max_acl * input_model.baseline_accuracy]
            dependent_var, independent_var = extract_var_regr(
                layer_df,
                layer_list,
                selected_data,
                var_select)
            print(f"extracted {len(dependent_var)} samples for  {var_select}")
            if len(dependent_var):
                extract_metric = extract_inertia if var_select == "inertia" else extract_acl
                # Evaluate model goodness
                score = evaluate_regression_model(dependent_var, independent_var)[0]
                print(f"Regression model with {var_select} score {score :.2%}, filter {max_acl * ratio:.2%}")
            ratio -= .1

        assert score > 0.7, f"Regression model not good enough"

        # Train final Prediction model
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(independent_var, dependent_var)
        assert (len(W) == len(
            linear_regression_model.coef_)), f"len(W)={len(W)}, len(linear_regression_model.coef_)={len(linear_regression_model.coef_)}"

        def predict_accuracy(x):
            # Helpers function to get inertia from layers k_id and predict result
            def predict_k_vect(id_vect):
                return linear_regression_model.predict(extract_metric(id_vect).reshape(-1, 1).transpose())

            metrics = np.apply_along_axis(predict_k_vect, arr=x, axis=1)[:, 0]
            return metrics
    else:
        def compute_accuracy(x):
            # Evaluate samples
            evaluation_df = input_model.score_approximation_list(np.apply_along_axis(extract_k, arr=x, axis=1),
                                                                 ds_scale=ds_scale, verbose=False, origin_str="nsga2")
            return evaluation_df.accuracy_loss.values

    class WeightSharing(Problem):

        def __init__(self):
            super().__init__(
                n_var=len(W),
                n_obj=2,
                # n_constr=1,
                xl=[0 for _ in layer_list],
                xu=[int(len(k_list[layer_name]) - 1) for layer_name in layer_list],
            )

        def _evaluate(self, X, out, *args, **kwargs):
            # Objective evaluation
            f1 = predict_accuracy(X) if not real_evaluation else compute_accuracy(X)
            f2 = calc_compression_rate_population(X, W, B)

            # Constraints evaluation
            g1 = f1 - (30 / 100)
            # g2 = [x[:, i] - layers_size for i in range(x.layer_shape[1])]

            out["F"] = np.column_stack([f1, f2])
            # out["G"] = np.column_stack([g1])

    pop_size = 100
    n_gen = 100
    print("population:{}, n_gen:{}, total: {}".format(pop_size, n_gen, pop_size * n_gen))

    method = get_algorithm(
        "nsga2",
        pop_size=pop_size,
        sampling=sampling if not isinstance(sampling, type(None)) else get_sampling("int_random"),
        crossover=get_crossover("int_sbx"),
        mutation=get_mutation("int_pm"),
        eliminate_duplicates=True,
    )

    problem = WeightSharing()

    if time_bounded:
        termination = ("time", "01:00:00")
    else:
        termination = ('n_gen', n_gen)

    # Explore DSE using problem and algo
    res = minimize(
        problem,
        method,
        termination=termination,
        save_history=True,
        verbose=True
    )

    # Log
    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)
    print("time: %s" % res.exec_time)

    k_vector = [extract_k(id_vect) for id_vect in res.X]

    # Evaluate samples
    population_df = input_model.score_approximation_list(k_vector, ds_scale=ds_scale, verbose=False, origin_str="nsga2",
                                                         h5_path=h5_path)

    # for each algorithm object in the history
    history_path = os.path.join(log_file, f"history_expansion_{'real' if real_evaluation else 'regr'}.h5")
    print(f"Save history @ {history_path}")

    # history
    history = None
    for i, entry in enumerate(tqdm(res.history)):
        x_values = entry.pop.get("X")
        f_values = entry.pop.get("F")
        for f, x in zip(f_values, x_values):
            history = log.append_data(
                history,
                {
                    "k_vector": [list(extract_k(x))],
                    "accuracy_loss": f[0],
                    "compression_rate": -f[1],
                    "iteration": entry.n_gen,
                })
    history.to_hdf(history_path, key="data")

    return population_df


def extract_var_regr(layer_df, layer_list, selected_data, var_select):
    from ast import literal_eval
    # Extract dependent var
    dependent_var = selected_data["accuracy_loss"].values
    # Extract independent var
    if var_select == "inertia":
        independent_var = np.array(
            [literal_eval(i) if isinstance(i, str) else i for i in selected_data["inertia_vector"].values])
    else:
        independent_var_k = np.array(
            [literal_eval(i) if isinstance(i, str) else i for i in selected_data["k_vector"].values])
        independent_var = np.array(
            [layer_df[layer_df.layer == layer].set_index('k').loc[independent_var_k.transpose()[i]].accuracy_loss for
             i, layer in enumerate(layer_list)]).transpose()
    return dependent_var, independent_var


def evaluate_regression_model(dependent_var, independent_var):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    independent_var = [literal_eval(i) if isinstance(i, str) else i for i in independent_var]
    x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, test_size=0.2,
                                                        random_state=0)
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(x_train, y_train)
    r_square = linear_regression_model.score(x_test, y_test)
    mse = mean_squared_error(y_test, linear_regression_model.predict(x_test))
    return r_square, np.sqrt(mse)
