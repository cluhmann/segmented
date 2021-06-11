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
        self.betas = None
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

        if self.data is None:
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
            elif "Intercept" in spec:
                raise ValueError(
                    "Received an invalid model specification.  Intercepts currently only permitted in the first model segment."
                )
            else:
                self.models += [spec]

        # extract data in accordance with specification
        # extract design matrices for various model components
        y_dmat = patsy.dmatrix(self.outcome_var, self.data)
        x_1_dmat = patsy.dmatrix(self.models[0], self.data)
        x_2_dmat = patsy.dmatrix(self.models[1], self.data)

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
        if changepoints is None:
            self.changepoints = None
            return

        if not(len(changepoints) == 1):
            raise ValueError(
                "Only a single changepoint can be modeled currently."
            )
        if self.data is None:
            raise ValueError(
                "Cannot set changepoints without valid data."
            )
        if "~" in spec:
            raise ValueError(
                "Received an invalid changepoint specification.  Changepoints may not specify an outcome variable."
            )
        cp_dmat = patsy.dmatrix(changepoints[0], self.data)
        # this is the name of the column in data that
        # represents our single covariate (changepoint predictor)
        self.changepoint_predictor_var_name = cp_dmat.design_info.column_names[0]
        self.changepoint_predictor_var = cp_dmat[:, 0]

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
        if self.changepoints is not None:
            if not isinstance(self.changepoints, list):
                raise ValueError(
                    "Received an invalid changepoints object.  Changepoints must be a list of patsy strings."
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
        if isinstance(self.changepoints, list) and (self.num_segments is not None):
            if len(self.changepoints) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the specified number of segments."
                )
        if isinstance(self.changepoints, list) and isinstance(self.models, list):
            if len(self.changepoints) != len(self.models):
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the number implied y the model specification."
                )

        # if number of segments is implied but not set, set it
        if self.num_segments is None:
            self.num_segments = len(self.models)

    def fit(self, x0):
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
        )

        self.store_parameters()

        return self

    def store_parameters(self):
        # parameters when n_segments = 2
        # b_0, b_1, b_2
        # and
        # t_1, t_2

        # t_1, b_0, b_1
        # t_2, b_2

        self.nodes = [
            self.predictor_var.min(),
            self.result.lowest_optimization_result.x[0],
        ]
        self.betas = [
            self.result.lowest_optimization_result.x[1],
            self.result.lowest_optimization_result.x[2],
            self.result.lowest_optimization_result.x[3]
        ]

        return self

    def predict(self, data, params=None):

        # prepare variables for model parameters
        if self.result is None:
            t_2, beta_0, beta_1, beta_2, error_sd = params
            # here, we define t_1, the first node, to be the
            # minimum of the data **passed when object was created**
            # NOT on the data we are now using for prediction
            t_1 = self.data[self.predictor_var_name].min()
        else:
            (
                beta_0,
                beta_1,
                beta_2
            ) = self.betas
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
