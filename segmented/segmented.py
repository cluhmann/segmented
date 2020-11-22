import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import patsy


class segmented:
    def __init__(self, models, changepoints=None, num_segments=None, data=None):

        self.models = None
        self.changepoints = None
        self.num_segments = None
        self.data = None
        self.result = None
        self.par_x = None

        self.set_models(models)
        self.set_changepoints(changepoints)
        self.set_num_segments(num_segments)
        self.set_data(data)

    def set_data(self, data=None):

        self.data = data

        # validate
        self.validate_parameters()

    def set_models(self, models=None):

        self.models = models

        # validate
        self.validate_parameters()

    def set_changepoints(self, changepoints=None):

        self.changepoints = changepoints

        # validate
        self.validate_parameters()

    def set_num_segments(self, num_segments):

        self.num_segments = num_segments

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
            if not (isinstance(self.models, list) or isinstance(self.models, string)):
                raise ValueError(
                    "Received an invalid models object.  Models must be a patsy string or a list of such strings."
                )
        if self.changepoints is not None:
            if not (
                isinstance(self.changepoints, list)
                or isinstance(self.changepoints, string)
            ):
                raise ValueError(
                    "Received an invalid changepoints object.  Changepoints must be a patsy string or a list of such strings."
                )
        if self.num_segments is not None:
            if not isinstance(self.num_segments, int):
                raise ValueError(
                    "Received an invalid num_segments object.  Number of segments must be an integer."
                )

        # check for conflicts among self.num_segments, self.models, and self.num_segments
        if isinstance(self.changepoints, list) and (self.num_segments is not None):
            if len(self.changepoints) != self.num_segments - 1:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the specified number of segments."
                )
        if isinstance(self.changepoints, list) and isinstance(self.models, list):
            if len(self.changepoints) != len(self.models) - 1:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the model specification."
                )
        if isinstance(self.models, list) and (self.num_segments is not None):
            if len(self.models) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by model specification conflicts with the specified number of segments."
                )

        # if number of segments is implied but not set, set it
        if self.num_segments is None:
            if isinstance(self.models, list):
                self.num_segments = len(self.models)
            elif isinstance(self.changepoints, list):
                self.num_segments = len(self.changepoints)
            else:
                raise ValueError("Number of segments must be specified.")

        # convert raw patsy strings to lists of patsy strings
        if isinstance(self.models, str):
            self.models = self.num_segments * [self.models]
        if isinstance(self.changepoints, str):
            self.changepoints = self.num_segments * [self.changepoints]

    def fit(self, x0):
        def logp(params, df):

            # parameters in general case
            # b_1 ... b_n
            # and
            # t_2 ... t_n
            # or
            # g_20... g_2n
            # g_30... g_3n
            # g_m0... g_mn

            # parameters when n_segments = 2
            # b_0, b_1, b_2
            # and
            # t_2
            # or
            # g_20, g_21

            beta_0, beta_1, beta_2, t_2, error_sd = params
            t_1 = None

            # extract design matrices for segment #1
            y_1_dmat, x_1_dmat = patsy.dmatrices(self.models[0], self.data)

            # make sure there is a single predictor
            # and no intercept (incercept is handled automatically)
            if not (len(x_1_dmat.design_info.column_names) == 1):
                raise ValueError(
                    "Received an invalid model specification.  Model must contain exactly one predictor variable."
                )

            # extract design matrices for segment #2
            y_2_dmat, x_2_dmat = patsy.dmatrices(self.models[1], self.data)

            # make sure there is a single predictor
            # and no intercept (incercept is handled automatically)
            if not (len(x_2_dmat.design_info.column_names) == 1):
                raise ValueError(
                    "Received an invalid model specification.  Model must contain exactly one predictor variable."
                )

            # check to make sure that the predictor variable is
            # consistent across model specifications
            if not (
                x_1_dmat.design_info.column_names[0]
                == x_2_dmat.design_info.column_names[0]
            ):
                raise ValueError(
                    "Received an invalid model specification.  Specification for each segment must contain same outcome variable."
                )

            # this is the name of the column in self.data
            # that represents our single predictor
            par_x_name = x_1_dmat.design_info.column_names[0]
            par_x_index = x_1_dmat.design_info.column_name_indexes[par_x_name]

            t_1 = x_1_dmat[par_x_index].min()
            x_1 = x_1_dmat[0] - t_1
            x_2 = x_2_dmat[0] - t_2

            y_1 = beta_0 + beta_1 * x_1
            y_2 = beta_2 * x_2

            y_hat = np.piecewise(
                x_1_dmat[par_x_index],
                [x_1_dmat[par_x_index] <= t_2, x_1_dmat[par_x_index] > t_2,],
                [y_1, y_1 + y_2],
            )

            p = scipy.stats.norm.pdf(y_1_dmat[0], y_hat, error_sd)
            return -1 * np.sum(np.log(p))

        self.result = scipy.optimize.minimize(
            logp, x0, args=(self.data,), method="Nelder-Mead", options={"maxiter": 1000}
        )

    def summary(self):
        # raise NotImplementedError("segmented.summary() not implemented at this time.")
        return self.result
