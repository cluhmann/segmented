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

        if len(models) > 2:
            raise NotImplementedError(
                    'segmented currently supports a maximum of 2 segments'
            )

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

        # deal with first model specification
        # extract outcome (to the left of "~")
        self.outcome_var_name, model0 = models[0].split("~")
        # here we need to explicitly exclude an intercept to just grab
        # the outcome column
        self.outcome_dmatrix = patsy.dmatrix('0+' + self.outcome_var_name, self.data)

        # extract first segment specification (to the right of "~")
        self.segment_specifications = [model0]
        self.segment_dmatrices = [patsy.dmatrix(model0, self.data)]

        # make sure there is an intercept in the first segment
        if "Intercept" not in self.segment_dmatrices[0].design_info.column_names:
            raise ValueError(
                "Received an invalid model specification.  Intercepts currently required in the first model segment."
            )

        # make sure there is an intercept in the first segment
        if not (self.segment_dmatrices[0].shape[1] == 2):
            raise ValueError(
                "Received an invalid model specification.  First model segment must include only a single predictor."
            )

        # deal with remaining model specifications
        predictor_name = list(set(self.segment_dmatrices[0].design_info.column_names) - set(["Intercept"]))
        for spec in models[1:]:
            if "~" in spec:
                raise ValueError(
                    "Received an invalid model specification.  Only the first entry in models may specify outcome variable."
                )
            self.segment_specifications += [spec]
            self.segment_dmatrices += [patsy.dmatrix(spec, self.data)]

            # make sure there is a single predictor
            # and no intercept (incercept is handled automatically for now)
            if not (len(self.segment_dmatrices[-1].design_info.column_names) == 1):
                raise ValueError(
                    "Received an invalid model specification.  Segments (other than the first) must omit an intercept and specify exactly one predictor variable."
                )

            # make sure predictor variable is identical across specifications
            if (not self.segment_dmatrices[-1].design_info.column_names[0] in predictor_name):
                raise ValueError(
                    "Received an invalid model specification.  Predictor variable must agree across segment specifications."
                )

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
                if len(changepoints) > 1:
                    raise ValueError(
                        "Only a single changepoint may be specified currently."
                    )
                if self.data is None:
                    raise ValueError(
                        "Cannot specify changepoints without valid data."
                    )
                if "~" in changepoints[0]:
                    raise ValueError(
                        "Received an invalid changepoint specification.  Changepoints may not specify an outcome variable."
                    )

                self.changepoint_dmatrices = [patsy.dmatrix(changepoints[0], self.data)]

            else:
                raise ValueError(
                    "Changepoints must be patsy strings."
                )

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
            self.num_segments = len(self.segment_specifications)

    def fit(self, guesses):

        # we got parameter values, but already results in self.result
        # warn user and use the parameter values that were passed in
        if (self.result is not None) and (params is not None) and (not np.array_equal(self.result, params, equal_nan=True)):
            warnings.warn('WARNING: segmented.predict() was previously fit, but received parameter values. Using parameter values passed to predict().', RuntimeWarning, stacklevel=2)

        if self.nodes_parametric:
            self._fit_parametric(guesses)
        else:
            self._fit_nonparametric(guesses)

    def _fit_nonparametric(self, changepoints):

        assert(len(changepoints) == (self.num_segments-1) )
        warnings.warn('WARNING: segmented.fit() running with unvalidated parameter guesses (x0).', RuntimeWarning, stacklevel=2)

        # the algorithm below assumes limited model specification
        # so we pull the individual predictors out of the design matrix
        y = self.outcome_dmatrix.reshape(-1)
        intercept = self.segment_dmatrices[0][:,0]
        x = self.segment_dmatrices[0][:,1]

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

            predictors = np.array([intercept, x] + U + V)

            result = np.linalg.lstsq(predictors.transpose(), y, rcond=None)

            beta = result[0][2:2+len(changepoints)]
            gamma = result[0][2+len(changepoints):]
            changepoints = changepoints - (gamma/beta)

            # check for convergence
            converged = np.abs(np.max(gamma)) < threshold

        # save results
        self.nodes = np.hstack([x.min(), changepoints[:]])
        self.coefs = result[0][0:2+len(changepoints)]
        # augment result so that initial node is at x=min(x)
        self.coefs[0] = self.coefs[0] + (self.coefs[1] * x.min())


    def _fit_parametric(self, x0):
        warnings.warn('WARNING: segmented.fit() running with unvalidated parameter guesses (x0).', RuntimeWarning, stacklevel=2)
        def logp(params, df):

            y_hat = self.predict(self.data, params=params)

            # likelihood of each observation
            y_dmat = self.data[self.outcome_var_name].to_numpy()
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
        # all of this assumes a first segment with an intercept and 1 predictor
        # and other segments that only include (the same) single predictor
        self.coefs = []
        param_index = 0
        for seg in self.segment_dmatrices:
            for _ in range(seg.shape[1]):
                self.coefs += [self.result.x[param_index]]
                param_index += 1

        predictor_name = list(set(self.segment_dmatrices[0].design_info.column_names) - set(["Intercept"]))
        self.nodes = [np.min(self.data[predictor_name].to_numpy())]

        for seg in self.changepoint_dmatrices:
            for __ in range(seg.shape[1]):
                self.nodes += [self.result.x[param_index]]
                param_index += 1

        return self


    def predict(self, data, params=None):

        if self.nodes_parametric:
            return self._predict_parametric(data, params)
        else:
            return self._predict_nonparametric(data, params)


    def _predict_nonparametric(self, data, params=None):

        predictor_name = list(set(self.segment_dmatrices[0].design_info.column_names) - set(["Intercept"]))

        # use the parameter values that were passed in
        if params is not None:
            if not(len(params) == 5):
                raise ValueError(
                    "Received invalid initial parameter value guesses.  Expected 5, received " + str(len(params)) +"."
                )


            t_2, beta_0, beta_1, beta_2, error_sd = params
            # here, we define t_1, the first node, to be the
            # minimum of the data **passed when object was created**
            # NOT on the data we are now using for prediction
            # otherwise, the other parameter values make no sense
            t_1 = self.data[predictor_var_name].min()
        else:
            # extract parameter values from self
            (
                beta_0,
                beta_1,
                beta_2
            ) = self.coefs
            t_1, t_2 = self.nodes

        x_1 = data[predictor_var_name] - t_1
        x_2 = data[predictor_var_name] - t_2

        # !!!
        # intercept is currently hardcoded into 1st segment
        # and omitted from second segment
        # !!!
        y_1 = beta_0 + beta_1 * x_1
        y_2 = beta_2 * x_2

        y_hat = np.where(data[predictor_var_name] <= t_2, y_1, y_1 + y_2)

        if self.result is not None:
            print(y_1)
            print(y_2)
            print(t_2)
            print(y_hat)

        return y_hat


    def _predict_parametric(self, data, params=None):

        # use the parameter values that were passed in
        if params is not None:
            params = list(params)
        else:
            params = self.coefs + self.nodes

        # here, we define t_1, the first node, to be the
        # minimum of the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # otherwise, the other parameter values make no sense
        if (self.result is not None) and (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: segmented.predict() was previously fit, but received data. Using previous initial node (e.g., min(x)) to predict().', RuntimeWarning, stacklevel=2)
        predictor_name = list(set(self.segment_dmatrices[0].design_info.column_names) - set(["Intercept"]))

        segments_params = []
        cp_params = [np.min(data[predictor_name].to_numpy())]
        for dmat in self.segment_dmatrices:
            temp_params = []
            for _ in range(dmat.shape[1]):
                asdf = params.pop()
                temp_params += [asdf]
                #temp_params += params.pop()
            segments_params += [np.array(temp_params)]
        for dmat in self.changepoint_dmatrices:
            for _ in range(dmat.shape[1]):
                cp_params += [params.pop()]
        cp_params = np.array(cp_params)

        # this params is not needed here, but empties the list
        # which permits the validation below
        error_sd = params.pop()

        if len(params) > 0:
            exp_num_params = sum(len(x) for x in segments_params)+ (len(cp_params)-1) + 1
            raise ValueError(
                "Received invalid initial parameter value guesses.  Expected " +str(exp_num_params) +", received " + str(exp_num_params+len(params)) +"."
            )

        x = data[predictor_name].to_numpy()
        x = x.astype(float)
        y_hat = np.zeros_like(x)

        for segment in range(self.num_segments):
            # broken down for debugging
            a = data[predictor_name]
            aprime = cp_params[segment]
            b = segments_params[segment].T
            c = x - cp_params[segment]
            d = b * c
            e = np.sum(d, axis=1)
            e = e.reshape([-1,1])
            y_hat += np.where(a <= aprime, np.zeros_like(data[predictor_name]), e)

        return y_hat


    def summary(self):
        # raise NotImplementedError("segmented.summary() not implemented at this time.")
        return self.result
