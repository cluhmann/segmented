import warnings 

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import patsy

tol = 1e-6


class segmented:
    """
    Class implementing segmented regression.

    ...

    Attributes
    ----------
    models : list
        List of model specifications in patsy format.

    data : Pandas Dataframe
        Data to be modeled.

    num_segments: int
        Number of segments to be modeled.

    outcome_var : str
        Column from self.data that represents.

    Methods
    -------
    set_models(models, validate=False)
        Set the segment specification(s).

    set_data(data, validate=False)
        Set the data to be modeled.

    set_num_segments(num_segments, validate=False)
        Set the number of segments.

    validate_parameters()
        Validate the object's current parameter values.

    fit()
        Estimate model parameters.

    summary()
        Provide information regarding the parameter estimation procedure.
    """

    def __init__(self, models, changepoints=None, data=None):

        self.num_segments = None
        self.data = None

        self.models = None
        self.outcome_var_name = None
        self.outcome_var = None
        self.predictor_var_name = None
        self.predictor_var = None

        self.nodes = None
        self.coefs = None
        self.result = None

        # data must be set before models
        self.set_data(data, validate=False)
        self.set_models(models, validate=False)
        self.set_changepoints(changepoints, validate=False)
        self.validate_parameters()

    def set_data(self, data=None, validate=True):
        self.data = data

        if validate:
            # validate
            self.validate_parameters()

    def set_models(self, models=None, validate=True):

        # we need valid data to parse the models
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                "Cannot set models without valid data."
            )

        # check first segment model
        if "~" not in models[0]:
            raise ValueError(
                "Received an invalid model specification.  First entry in models must specify outcome variable."
            )
        else:
            self.outcome_var_name, model0 = models[0].split("~")
            self.outcome_var = self.data[self.outcome_var_name]
            self.models = [model0]

        # deal with remaining model specifications
        for spec in models[1:]:
            if "~" in spec:
                raise ValueError(
                    "Received an invalid model specification.  Only the first entry in models may specify outcome variable."
                )
            else:
                self.models += [spec]

        if len(self.models) > 2:
            raise NotImplementedError(
                    'segmented currently supports a maximum of 2 segments'
            )

        # extract data in accordance with specification
        # extract design matrices for various model components
        y_dmat = patsy.dmatrix(self.outcome_var, self.data)
        x_1_dmat = patsy.dmatrix(self.models[0], self.data)
        x_2_dmat = patsy.dmatrix(self.models[1], self.data)

        # make sure there is an intercept in the first segment
        if "Intercept" not in x_1_dmat.design_info.column_names:
            raise ValueError(
                "Received an invalid model specification.  Intercepts currently required in the first model segment."
            )
        # make sure there is a single predictor
        # and no intercept (incercept is handled automatically for now)
        if not (len(x_2_dmat.design_info.column_names) == 1):
            raise ValueError(
                "Received an invalid model specification.  Segments (other than the first) must omit an intercept and specify exactly one predictor variable."
            )

        # make sure predictor variable is identical across specifications
        if (
            not x_2_dmat.design_info.column_names[0]
            in x_1_dmat.design_info.column_names
        ):
            raise ValueError(
                "Received an invalid model specification.  Predictor variable must agree across segment specifications."
            )

        # this is the name of the column in data
        # that represents our single predictor
        self.predictor_var_name = x_2_dmat.design_info.column_names[0]
        self.predictor_var = x_2_dmat[:, 0]

        if validate:
            # validate
            self.validate_parameters()

    def set_changepoints(self, changepoints, validate=True):
        self.nodes_parametric = False

        if changepoints is None:
            self.nodes = None
            return

        # if we have received a list
        if isinstance(changepoints, list):

            # if we have received parametric node placement specifications
            if isinstance(changepoints[0], str):
                self.nodes_parametric = True
                raise ValueError(
                    "Parametric node placement is not currently supported."
                )
                if not(len(changepoints) == 1):
                    raise ValueError(
                        "Only a single changepoint can be modeled currently."
                    )

                if "~" in changepoints:
                    raise ValueError(
                        "Received an invalid changepoint specification.  Changepoints may not specify an outcome variable."
                    )
                if self.data is None:
                    raise ValueError(
                        "Cannot set changepoints without valid data."
                    )
                cp_dmat = patsy.dmatrix(changepoints[0], self.data)
                # this is the name of the column in data that
                # represents our single covariate (changepoint predictor)
                self.node_predictor_var_name = cp_dmat.design_info.column_names[0]
                self.node_predictor_var = cp_dmat[:, 0]

        if validate:
            # validate
            self.validate_parameters()

    def set_num_segments(self, num_segments, validate=True):

        self.num_segments = num_segments

        if validate:
            # validate
            self.validate_parameters()

    def validate_parameters(self):

        # type checking
        if self.data is not None:
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "Received an invalid data object.  Data must be a pandas dataframe."
                )
        if self.models is not None:
            if not isinstance(self.models, list):
                raise ValueError(
                    "Received an invalid models object.  Models must be a list of patsy strings."
                )
        if self.nodes is not None:
            if not isinstance(self.nodes, list):
                raise ValueError(
                    "Received an invalid changepoints object.  Changepoints must be a list of initial guesses or patsy strings."
                )
        if self.num_segments is not None:
            if not isinstance(self.num_segments, int):
                raise ValueError(
                    "Received an invalid num_segments object.  Number of segments must be an integer."
                )

        # check for conflicts among self.num_segments and self.models
        if isinstance(self.models, list) and (self.num_segments is not None):
            if len(self.models) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by model specification conflicts with the specified number of segments."
                )
        if isinstance(self.nodes, list) and (self.num_segments is not None):
            if len(self.nodes) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the specified number of segments."
                )
        if isinstance(self.nodes, list) and isinstance(self.models, list):
            if len(self.nodes) != len(self.models):
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the number implied y the model specification."
                )

        # if number of segments is implied but not set, set it
        if self.num_segments is None:
            self.num_segments = len(self.models)

    def fit(self, guesses):

        if self.nodes_parametric:
            self._fit_parametric(guesses)
        else:
            self._fit_nonparametric(guesses)

    def _fit_nonparametric(self, changepoints):

        assert(len(changepoints) == (self.num_segments-1) )

        x = self.predictor_var
        y = self.outcome_var

        # this is based on the method described in
        # Muggeo (2003, Statist. Med.)
        threshold = .00001 * np.min(np.diff(x))
        converged = False
        while not converged:
            U = [np.clip(x - changepoints[0], 0, None)]
            V = [(x - changepoints[0]) > 0]
            for changepoint in changepoints[1:]:
                U += [np.clip(x - changepoint, 0, None)]
                V += [(x - changepoint) > 0]

            predictors = np.array([np.ones_like(x), x] + U + V)

            result = np.linalg.lstsq(predictors.transpose(), y, rcond=None)
            beta = result[0][2:2+len(changepoints)]
            gamma = result[0][2+len(changepoints):]
            changepoints = changepoints - (gamma/beta)

            converged = np.abs(np.max(gamma)) < threshold

        # save results
        self.nodes = [x.min(), changepoints[:]]
        self.coefs = result[0][0:2+len(changepoints)]
        # augment result so that initial node is at x=min(x)
        self.coefs[0] = self.coefs[0] + (self.coefs[1] * x.min())


    def _fit_parametric(self, x0):
        warnings.warn('WARNING: segmented.fit() running with unvalidated parameter guesses (x0).', RuntimeWarning)
        def logp(params, df):

            y_hat = self.predict(self.data, params=params)

            # likelihood of each observation
            y_dmat = self.data[self.outcome_var_name]
            error_sd = params[-1]
            p = scipy.stats.norm.pdf(y_dmat, y_hat, error_sd)

            # log likelihood of entire data set
            return -1 * np.sum(np.log(np.clip(p, tol, 1 - tol)))

        # self.result = scipy.optimize.minimize(
        #    logp, x0, args=(self.data,), method="BFGS", options={"maxiter": 1000}
        # )
        self.result = scipy.optimize.basinhopping(
            logp,
            x0,
            minimizer_kwargs={"args": (self.data,), "method": "BFGS"},
            niter=500,
            T = 100,
            stepsize = 4,
        ).lowest_optimization_result

        # save results
        self.nodes = [
            self.predictor_var.min(),
            self.result.x[0],
        ]
        self.coefs = [
            self.result.x[1],
            self.result.x[2],
            self.result.x[3]
        ]

        return self

    def predict(self, data, params=None):

        # we got parameter values, but already results in self.result
        # warn user and use the parameter values that were passed in
        if (self.result is not None) and (params is not None) and (not np.array_equal(self.result, params, equal_nan=True)):
            warnings.warn('WARNING: segmented.predict() was previously fit, but received parameter values. Using parameter values passed to predict().', RuntimeWarning)
        # use the parameter values that were passed in
        if params is not None:
            t_2, beta_0, beta_1, beta_2, error_sd = params
            # here, we define t_1, the first node, to be the
            # minimum of the data **passed when object was created**
            # NOT on the data we are now using for prediction
            # otherwise, the other parameter values make no sense
            t_1 = self.data[self.predictor_var_name].min()
        else:
            # extract parameter values from self
            (
                beta_0,
                beta_1,
                beta_2
            ) = self.coefs
            t_1, t_2 = self.nodes

        x_1 = data[self.predictor_var_name] - t_1
        x_2 = data[self.predictor_var_name] - t_2

        # !!!
        # intercept is currently hardcoded into 1st segment
        # and omitted form second segment
        # !!!
        y_1 = beta_0 + beta_1 * x_1
        y_2 = beta_2 * x_2

        y_hat = np.where(data[self.predictor_var_name] <= t_2, y_1, y_1 + y_2)

        if self.result is not None:
            print(y_1)
            print(y_2)
            print(t_2)
            print(y_hat)

        return y_hat


    def summary(self):
        # raise NotImplementedError("segmented.summary() not implemented at this time.")
        return self.result
