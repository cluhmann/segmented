import warnings 

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import patsy

from multiprocessing import Pool
from contextlib import nullcontext



tol = 1e-6



class bayes_base:

    def __init__(self):

        self.data = None

        self.num_segments = None

        self.y_var = None
        self.x_var = None

        self.outcome_dmatrix = None

        self.segment_specifications = None
        self.segment_dmatrices = None

        self.changepoint_specifications = None
        self.changepoint_dmatrices = None

        self.changepoint_params = None
        self.segment_params = None

        self.priors = None
        self.trace = None


    def set_data(self, data=None, validate=True):
        self.data = data

        if validate:
            # validate
            self.validate_parameters()


    def set_segments(self, specs=None, x_var=None, validate=True):

        self.x_var = x_var
        self.segment_specifications = specs

        # we need valid data to parse the models
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                "Cannot set segments without valid data."
            )

        # check first segment model
        if "~" not in specs[0]:
            raise ValueError(
                "Received an invalid model specification.  First segment specification must include outcome variable."
            )

        # deal with first model specification
        self.outcome_dmatrix,dmat0 = patsy.dmatrices(specs[0], self.data, return_type='dataframe')
        self.y_var = self.outcome_dmatrix.columns[0]
        self.segment_dmatrices = [dmat0]

        # ensure there is a single predictor in the first segment
        # or that the x_var has been explicitly passed in
        if (self.segment_dmatrices[0].shape[1] > 2) and (self.x_var is not None):
            raise ValueError(
                "Received an invalid model specification.  First model segment must include only a single predictor unless x_var is set."
            )

        # set the x_var now that we know what it is
        if self.x_var is None:
            self.x_var = list(set(self.segment_dmatrices[0].columns) - set(["Intercept"]))

        # deal with remaining model specifications
        for spec in specs[1:]:
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

        if validate:
            # validate
            self.validate_parameters()


    def set_changepoints(self, changepoints=None, validate=True):
        # if we have received a list
        if isinstance(changepoints, list):

            # if we have received changepoint specifications
            if isinstance(changepoints[0], str):
                if self.data is None:
                    raise ValueError(
                        "Cannot specify changepoints without valid data."
                    )
                self.changepoint_specifications = []
                self.changepoint_dmatrices = []
                for spec in changepoints:
                    if "~" in spec:
                        raise ValueError(
                            "Received an invalid changepoint specification.  Changepoints may not specify an outcome variable."
                        )
                    self.changepoint_specifications += [spec]
                    self.changepoint_dmatrices += [patsy.dmatrix(spec, self.data, return_type='dataframe')]

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
        if self.segment_specifications is not None:
            if not isinstance(self.segment_specifications, list):
                raise ValueError(
                    "Received an invalid models object.  Models must be a list of patsy strings."
                )
        if self.changepoint_params is not None:
            if not isinstance(self.changepoint_coefs, list):
                raise ValueError(
                    "Received an invalid changepoints object.  Changepoints must be a list of initial guesses or patsy strings."
                )
        if self.num_segments is not None:
            if not isinstance(self.num_segments, int):
                raise ValueError(
                    "Received an invalid num_segments object.  Number of segments must be an integer."
                )

        # check for conflicts among self.num_segments and self.segment_specifications
        if isinstance(self.segment_specifications, list) and (self.num_segments is not None):
            if len(self.segment_specifications) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by model specification conflicts with the specified number of segments."
                )
        if isinstance(self.changepoint_params, list) and (self.num_segments is not None):
            if len(self.changepoint_params) != self.num_segments:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the specified number of segments."
                )
        if isinstance(self.changepoint_params, list) and isinstance(self.segment_specifications, list):
            if len(self.changepoint_params) != len(self.segment_specifications):
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the number implied y the model specification."
                )

        # if number of segments is implied but not set, set it
        if self.num_segments is None:
            self.set_num_segments(len(self.segment_specifications))


    def summary(self):
        # raise NotImplementedError("segmented.summary() not implemented at this time.")
        return self.trace




class bayes(bayes_base):

    def __init__(self, segments, y_var=None, x_var=None, changepoints=None, data=None):

        super().__init__()

        # data must be set before models/changepoints
        self.set_data(data, validate=False)
        self.set_segments(segments, x_var=x_var, validate=False)
        self.set_changepoints(changepoints, validate=False)
        self.validate_parameters()


    def set_priors(self, priors):
        # TODO
        # the plan is to accept frozen scipy distributions
        # e.g., scipy.stats.norm(loc=10, scale=3.7)
        # the pdfs of these frozen distributions can then be used
        # to generate log priors at sampled locations during sampling
        pass


    def fit(self, num_samples=1000, num_burnin=200, num_walkers=30, num_cores=None):

        try:
            import emcee
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Bayesian estimation requires emcee to be installed."
            ) from e

        # figure out a good place to begin our sampling
        p0 = self.gen_start_point(num_walkers)

        # instantiate context appropriate for the number of cores requested
        if cores is not None:
            poolholder = Pool(processes=num_cores)
        else:
            poolholder = nullcontext()
        with poolholder as pool:
            # generate sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logp,
                                            moves=emcee.moves.DESnookerMove(),
                                            pool=pool)
            if num_burnin > 0:
                p0 = sampler.run_mcmc(p0, num_burnin, progress=True)
                sampler.reset()
            sampler.run_mcmc(p0, nsamples, progress=True)

        return sampler.get_chain()

    def gather_parameters(self):
        # for now, just generate strings of each specification parameter
        # and each changepoint parameter

        param_names = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            param_names += [f'sgmt_{var}_{i}' for var in dmat.columns]

        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            param_names += [f'cp_{var}_{i}' for var in dmat.columns]

        param_names += ['sigma_0']

        return param_names


    def gen_start_point(self, num_walkers):
        # set up the starting point for each parameter and walker
        num_sgmt_params = len([len(x) for x in self.segment_dmatrices.columns])
        num_cp_params = len([len(x) for x in self.changepoint_dmatrices.columns])
        # plus 1 because of the SD of the error term
        # at some point we need to have a list of error specifications
        num_dims = num_sgmt_params + num_cp_params + 1
        # initial coefficients for segment predictors
        # TODO: these should probably be informed by the priors
        p0 = [0 for i in range(num_sgmt_params)]
        # initial coefficients for changepoint predictors
        # intercept set to mean of x
        p0 += [np.mean(x)] + [0 for i in range(num_cp_params-1)]
        # generate a "fuzzed" version of those initial guesses
        # for each walker that we will use for MCMC
        jitter = .04
        p0 += np.random.normal(scale=jitter, size=(num_walkers, len(p0)))
        # intial guess for error term SD must be positive
        # we use the sample SD and then scale it by some factor
        sdfactor = 0.1
        p0 = np.hstack([p0, sdfactor * np.std(y) * np.random.random(num_walkers).reshape(-1,1)])
        return p0



    def logp(self, params):
        return self.log_prior(params) + self.likelihood(params)


    def log_prior(self, params, debug=False):
        # TODO
        pass


    def likelihood(self, params, debug=False):

        # helper function for fit()/optimization method below

        # predict outcome variable
        y_hat = self.predict(self.data, params=params, fitting=True, debug=debug)

        # likelihood of each observation
        y_dmat = self.outcome_dmatrix.to_numpy().reshape(-1)
        # error is currently normal and parameter is the log of the SD
        error_sd = np.exp(params[-1])
        logp = scipy.stats.norm.logpdf(y_dmat, y_hat, error_sd)

        if debug:
            print('fitting')
            print('params: '+str(params))
            print('y_dmat: '+str(y_dmat.reshape([-1,1])))
            print('y_hat: '+str(y_hat))
            print('logp: '+str(logp))
            print('sumlogp: '+str(np.sum(logp)))

        # negative log likelihood of entire data set
        return np.sum(logp)


    def predict(self, data, params=None, fitting=False, debug=False):

        # use the parameter values that were passed in
        if params is not None:
            params = list(params)
        else:
            params = self.segment_coefs + self.changepoint_coefs

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
        # TODO: we should store segment (and cp) parameters as a list of lists
        # so that the specifications, the dmatrices, and the parameters
        # are structured in the same way
        # e.g., params = [p for s in self.segment_coefs for p in s]
        segments_params = []
        # for each segment
        for dmat in self.segment_dmatrices:
            temp_params = []
            # for each predictor in specification
            for _ in range(dmat.shape[1]):
                if debug:
                    print('spopping ' + str(params[0]))
                # check for too few parameters passed in
                try:
                    p = params.pop(0)
                except IndexError:
                    raise ValueError(
                        "predict() received too few parameter values to unpack."
                    )
                temp_params += [p]
            segments_params += [np.array(temp_params)]

        # skip the first, dummy dmatrix
        # insert extract the first, min(x) node
        cp_params = [[np.min(self.data[predictor_name].to_numpy())]]
        # for second segment and on
        for dmat in self.changepoint_dmatrices[1:]:
            cp_params_current_segment = []
            # for each predictor in specification
            for _ in range(dmat.shape[1]):
                if debug:
                    print('cpopping ' + str(params[0]))
                cp_params_current_segment += [params.pop(0)]

            cp_params += [np.array(cp_params_current_segment)]

        # this param is not needed here, but should empty the list
        # which permits the subsequent validation
        error_logsd = params.pop(0)
        # the parameter list should now be empty
        # if not, something has gone wrong
        if len(params) > 0:
            exp_num_params = sum(len(x) for x in segments_params)+ (len(cp_params)-1) + 1
            raise ValueError(
                "Received invalid initial parameter value guesses.  Expected " +str(exp_num_params) +" values, received " + str(exp_num_params+len(params)) +"."
            )

        # extract primary predictor variable
        #print('pname: ' + str(predictor_name))
        x = patsy.dmatrix(predictor_name, data, return_type='dataframe')[predictor_name].to_numpy(dtype=float).reshape([-1,1])
        # reconstruct intercept vector
        intercept = np.ones_like(x) * self.segment_dmatrices[0]['Intercept'].to_numpy()[0]
        y_hat = np.zeros((x.shape[0],))
        zeros = np.zeros_like(x)

        # for each segment
        for segment in range(self.num_segments):
            # reconstruct vector of changepoints associated with segment #2
            cp = cp_params[segment] * patsy.dmatrix(self.changepoint_specifications[segment], data, return_type='dataframe')
            cp = cp.sum(axis=1).to_numpy(dtype=float).reshape([-1,1])

            if debug:
                print('cp_params: ' +str(cp_params))
                print(self.changepoint_specifications[segment])
                print(data)
                print('cp: ' + str(cp))
                print(cp.shape)

            # zero out predictor variable "left" of segment
            effective_x = x - cp
            effective_x = np.clip(effective_x, 0, None)
            # insert intercept if specified
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
                e = np.sum(segments_params[segment] * effective_x, axis=1).reshape([-1,1])

            # boolean indicating which datapoints are "in" current segment
            comp = data[predictor_name] < np.squeeze(cp)
            # mask segment predictions to only appropriate datapoints
            temp = np.where(comp, np.squeeze(zeros), np.squeeze(e))

            if debug:
                print('eshp'+str(e.shape))
                print('yhtshp'+str(y_hat.shape))
                print('cpshp'+str(cp.shape))
                print('cp: '+str(cp))
                print('cmpshp'+str(comp.shape))
                print('cmp: '+str(comp))
                print('e: ' +str(e))
                print('y_hatpreupdate: ' +str(y_hat))

            # add predictions to "running" totals
            y_hat = y_hat + temp

        return y_hat



