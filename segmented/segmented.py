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
        self.outcome_dmatrix = patsy.dmatrix('0+' + self.outcome_var_name, self.data, return_type='dataframe')

        # extract first segment specification (to the right of "~")
        self.segment_specifications = [model0]
        self.segment_dmatrices = [patsy.dmatrix(model0, self.data, return_type='dataframe')]

        # make sure there is an intercept in the first segment
        if "Intercept" not in self.segment_dmatrices[0].columns:
            raise ValueError(
                "Received an invalid model specification.  Intercepts currently required in the first model segment."
            )

        # make sure there is an intercept in the first segment
        if not (self.segment_dmatrices[0].shape[1] == 2):
            raise ValueError(
                "Received an invalid model specification.  First model segment must include only a single predictor."
            )

        # deal with remaining model specifications
        predictor_name = list(set(self.segment_dmatrices[0].columns) - set(["Intercept"]))
        for spec in models[1:]:
            if "~" in spec:
                raise ValueError(
                    "Received an invalid model specification.  Only the first entry in models may specify outcome variable."
                )
            self.segment_specifications += [spec]
            self.segment_dmatrices += [patsy.dmatrix(spec, self.data, return_type='dataframe')]

            # make sure there is a single predictor
            # and no intercept (incercept is handled automatically for now)
            if not (len(self.segment_dmatrices[-1].columns) == 1):
                raise ValueError(
                    "Received an invalid model specification.  Segments (other than the first) must omit an intercept and specify exactly one predictor variable."
                )

            # make sure predictor variable is identical across specifications
            if (not self.segment_dmatrices[-1].columns[0] in predictor_name):
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

                self.changepoint_specifications = ['1'] + changepoints
                self.changepoint_dmatrices = [
                    pd.DataFrame({'Intercept':np.ones(self.outcome_dmatrix.shape[1])}),
                    patsy.dmatrix(changepoints[0], self.data, return_type='dataframe')
                ]

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
        if (self.result is not None) and (guesses is not None) and (not np.array_equal(self.result, guesses, equal_nan=True)):
            warnings.warn('WARNING: segmented.predict() was previously fit, but received parameter values. Using parameter values passed to predict().', RuntimeWarning, stacklevel=2)

        # clear out any old results
        self.result = None

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


    def _fit_parametric(self, x0, debug=False):
        warnings.warn('WARNING: segmented.fit() running with unvalidated parameter guesses (x0).', RuntimeWarning, stacklevel=2)
        def logp(params, df):

            y_hat = self.predict(self.data, params=params, fitting=True, debug=debug)

            # likelihood of each observation
            y_dmat = self.data[self.outcome_var_name].to_numpy()
            error_sd = np.exp(params[-1])
            logp = scipy.stats.norm.logpdf(y_dmat.reshape([-1,1]), y_hat, error_sd)

            if debug:
                print('fitting')
                print('params: '+str(params))
                print('y_dmat.reshp: '+str(y_dmat.reshape([-1,1])))
                print('y_hat: '+str(y_hat))
                print('logp: '+str(logp))
                print('sumlogp: '+str(np.sum(logp)))

            # negative log likelihood of entire data set
            return -1 * np.sum(logp)

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
        self.coefs = []
        param_index = 0
        for seg in self.segment_dmatrices:
            for _ in range(seg.shape[1]):
                self.coefs += [self.result.x[param_index]]
                param_index += 1

        predictor_name = list(set(self.segment_dmatrices[0].columns) - set(["Intercept"]))
        self.nodes = [np.min(self.data[predictor_name].to_numpy())]

        for seg in self.changepoint_dmatrices:
            for __ in range(seg.shape[1]):
                self.nodes += [self.result.x[param_index]]
                param_index += 1

        return self


    def predict(self, data, params=None, fitting=False, debug=False):

        if self.nodes_parametric:
            return self._predict_parametric(data, params, fitting=fitting, debug=debug)
        else:
            return self._predict_nonparametric(data, params, fitting=fitting, debug=debug)


    def _predict_nonparametric(self, data, params=None, fitting=False, debug=False):

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

        return y_hat


    def _predict_parametric(self, data, params=None, fitting=False, debug=False):

        # use the parameter values that were passed in
        if params is not None:
            params = list(params)
        else:
            params = self.coefs + self.nodes

        if debug:
            print('params: ' + str(params))

        # here, we define t_1, the first node, to be the
        # minimum of the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # otherwise, the other parameter values make no sense
        if fitting and (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: segmented.predict() was previously fit, but received data. Using previous initial node (e.g., min(x)) to predict().', RuntimeWarning, stacklevel=2)

        # unpack the name of the main predictor (x)
        predictor_name = list(set(self.segment_dmatrices[0].columns) - set(["Intercept"]))
        # there should only be 1 predictor
        # and it should be common to all segments
        assert(len(predictor_name) == 1)
        predictor_name = predictor_name[0]

        # unpack the name of the covariable (z)
        covariable_name = list(set(self.changepoint_dmatrices[1].columns) - set(["Intercept"]))
        assert(len(covariable_name) == 1)
        covariable_name = covariable_name[0]

        # unpack parameters associated with each segment
        segments_params = []
        for dmat in self.segment_dmatrices:
            temp_params = []
            for _ in range(dmat.shape[1]):
                if debug:
                    print('spopping ' + str(params[0]))
                asdf = params.pop(0)
                temp_params += [asdf]
                #temp_params += params.pop(0)
            segments_params += [np.array(temp_params)]

        # skip the first, dummy dmat
        # insert extract the first, min(x) node
        cp_params = [[np.min(self.data[predictor_name].to_numpy())]]
        # and then unpack the rest
        for dmat in self.changepoint_dmatrices[1:]:
            cp_params_current_segment = []
            # we need to 
            for _ in range(dmat.shape[1]):
                if debug:
                    print('cpopping ' + str(params[0]))
                cp_params_current_segment += [params.pop(0)]

            cp_params += [np.array(cp_params_current_segment)]

        # this param is not needed here, but should empty the list
        # which permits the validation below
        error_sd = params.pop(0)
        # the parameter list should now be empty
        # if not, something has gone wrong
        if len(params) > 0:
            exp_num_params = sum(len(x) for x in segments_params)+ (len(cp_params)-1) + 1
            raise ValueError(
                "Received invalid initial parameter value guesses.  Expected " +str(exp_num_params) +" values, received " + str(exp_num_params+len(params)) +"."
            )

        # extract primary predictor variable
        print('pname: ' + str(predictor_name))
        x = patsy.dmatrix(predictor_name, data, return_type='dataframe')[predictor_name].to_numpy(dtype=float).reshape([-1,1])
        # reconstruct intercept vector
        intercept = np.ones_like(x) * self.segment_dmatrices[0]['Intercept'].to_numpy()[0]
        print('cname: ' + covariable_name)
        y_hat = np.zeros((x.shape[0],))
        zeros = np.zeros_like(x)

        print(x.shape)
        print(intercept.shape)
        print(y_hat.shape)
        print(zeros.shape)

        for segment in range(self.num_segments):
            # reconstruct vector of changepoints associated with segment #2
            cp = cp_params[segment] * patsy.dmatrix(self.changepoint_specifications[segment], data, return_type='dataframe')
            cp = cp.sum(axis=1).to_numpy(dtype=float).reshape([-1,1])

            print('cp_params: ' +str(cp_params))
            print(self.changepoint_specifications[segment])
            print(data)
            print('cp: ' + str(cp))
            print(cp.shape)

            # zero out predictor variable at left edge of segment
            effective_x = x - cp
            effective_x = np.clip(effective_x, 0, None)
            # insert intercept when appropriate
            if 'Intercept' in self.segment_dmatrices[segment].columns:
                effective_x = np.hstack([intercept,
                                        effective_x])
            else:
                # intercept not requested
                pass

            if debug:
                print('predicting segment #' +str(segment))
                print('segment params: ' +str(segments_params[segment]))
                print('cp_params: ' +str(cp_params[segment]))
                # broken down for debugging
                a = data[predictor_name]
                aprime = cp_params[segment]
                b = segments_params[segment]
                c = effective_x
                print('xs: ' +str(x))
                print('eff. xs: ' +str(c))
                d = b * c
                print('product: ' +str(d))
                f = np.sum(d, axis=1)
                print('sum: ' +str(f))
                e = f.reshape([-1,1])
                print('reshape: ' +str(e))
            else:
                e = np.sum(segments_params[segment] * effective_x.T, axis=1).reshape([-1,1])
            print('eshp'+str(e.shape))
            print('yhtshp'+str(y_hat.shape))
            print('cpshp'+str(cp.shape))
            print('cp: '+str(cp))
            comp = data[predictor_name] < np.squeeze(cp)
            print('cmpshp'+str(comp.shape))
            print('cmp: '+str(comp))
            temp = np.where(comp, np.squeeze(zeros), np.squeeze(e))
            print('e: ' +str(e))
            print('temp: ' +str(temp))
            print('y_hatpreupdate: ' +str(y_hat))
            y_hat = y_hat + temp
            print('y_hatpostupdate: ' +str(y_hat))

        return y_hat


    def summary(self):
        # raise NotImplementedError("segmented.summary() not implemented at this time.")
        return self.result
