import warnings 

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import patsy
import arviz

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

        self.family = None
        self.link = None


    def set_data(self, data=None, validate=True):
        self.data = data

        if validate:
            # validate
            self.validate_parameters()


    def set_segments(self, specs=None, y_var=None, x_var=None, validate=True):

        self.y_var = y_var
        self.x_var = x_var
        self.segment_specifications = specs

        # we need valid data to parse the models
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(
                "Cannot set segments without valid data."
            )

        # check first segment model
        if "~" not in specs[0]:
            if self.y_var is None:
                raise ValueError(
                    "Received an invalid model specification.  First segment specification must include outcome variable unless y_var is set."
                )
            else:
                # deal with first model specification
                self.outcome_dmatrix = patsy.dmatrix('0+'+self.y_var, self.data, return_type='dataframe')
                dmat0 = patsy.dmatrix(specs[0], self.data, return_type='dataframe')
                self.segment_dmatrices = [dmat0]
                self.segment_specifications = [specs[0]]

        else:
            # deal with first model specification
            self.outcome_dmatrix,dmat0 = patsy.dmatrices(specs[0], self.data, return_type='dataframe')
            self.y_var = self.outcome_dmatrix.columns[0]
            self.segment_dmatrices = [dmat0]
            self.segment_specifications = [specs[0]]


        # ensure there is a single predictor in the first segment
        # or that the x_var has been explicitly passed in
        if (self.segment_dmatrices[0].shape[1] > 2) and (self.x_var is None):
            raise ValueError(
                "Received an invalid model specification.  First model segment must include only a single predictor unless x_var is set."
            )

        # set the x_var now that we know what it is
        if self.x_var is None:
            self.x_var = list(set(self.segment_dmatrices[0].columns) - set(["Intercept"]))[0]

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
            #if not (len(self.segment_dmatrices[-1].columns) == 1):
            #    raise ValueError(
            #        "Received an invalid model specification.  Segments (other than the first) must omit an intercept and specify exactly one predictor variable."
            #    )

        # set the number of segments explicitly
        self.set_num_segments(len(self.segment_specifications), validate=validate)

        if validate:
            # validate
            self.validate_parameters()


    def set_changepoints(self, changepoints=None, validate=True):
        # if the changepoints are implicit, specify them explicitly here
        # as an "intercept-only" specification
        if changepoints is None:
            changepoints = ['1' for i in range(self.num_segments-1)]

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
                    dmat = patsy.dmatrix(spec, self.data, return_type='dataframe')
                    self.changepoint_dmatrices += [dmat]

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











class bayes(bayes_base):

    def __init__(self, segments, y_var=None, x_var=None, changepoints=None, priors=None, data=None):

        super().__init__()

        # data must be set before models/changepoints
        self.set_data(data, validate=False)
        self.set_segments(segments, y_var=y_var, x_var=x_var, validate=False)
        self.set_changepoints(changepoints, validate=False)
        self.set_priors(priors=priors)
        self.validate_parameters()


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
        if num_cores is not None:
            poolholder = Pool(processes=num_cores)
        else:
            poolholder = nullcontext()
        with poolholder as pool:
            # generate sampler
            print('number of priors is '+str(len(self.prior)))
            print('number of parameters is '+str(p0.shape[1]))
            sampler = emcee.EnsembleSampler(num_walkers, p0.shape[1], self.logp,
                                            moves=emcee.moves.DESnookerMove(),
                                            pool=pool)
            if num_burnin > 0:
                p0 = sampler.run_mcmc(p0, num_burnin, progress=True)
                sampler.reset()
            sampler.run_mcmc(p0, nsamples, progress=True)

        # define variable names, it cannot be inferred from emcee
        var_names = self.gather_parameter_names()
        return arviz.from_emcee(sampler, var_names=var_names)


    def gen_start_point(self, num_walkers):

        # set up the starting point for each parameter and walker
        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        # with the Dirichlet-prior, we need num_changepoints + 1 alphas
        # so add 1 to this running sum
        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        # plus 1 because of the SD of the error term
        # at some point we need to have a list of error specifications
        num_dims = num_sgmt_params + num_cp_params + 1
        # initial coefficients for segment predictors
        # TODO: these should probably be informed by/sampled from the prior
        p0 = [0 for i in range(num_sgmt_params)]
        # initial coefficients for changepoint predictors
        cp_guesses = np.linspace(.25, 1, num=num_cp_params, endpoint=False)
        # normalize them
        cp_guesses = cp_guesses / np.sum(cp_guesses)
        # insert them in the vector of parameters
        p0 += list(cp_guesses)
        # assume a single
        #p0 += [1 for i in range(num_cp_params-1)]
        # generate a "fuzzed" version of those initial guesses
        # for each walker that we will use for MCMC
        p0 = np.array(p0)
        print((num_walkers, len(p0)))
        jitter = .04
        p0 = p0 + np.random.normal(scale=jitter, size=(num_walkers, len(p0)))
        # intial guess for error term SD must be positive
        # we use the sample SD and then scale it by some factor
        sdfactor = 0.1
        p0 = np.hstack([p0, sdfactor * np.std(self.data[self.y_var]) * np.random.random(num_walkers).reshape(-1,1)])
        print('gen_start_point believes we have ' +str(p0.shape)+' parameters')
        print(str(num_sgmt_params)+' for the segments')
        print(str(num_cp_params)+' for the changepoints')
        print(str(len(cp_guesses))+' for the changepoints?')
        print('1 for the error')
        print('num_dims='+str(num_dims))
        return p0



    def gather_parameter_names(self):
        # for now, just generate strings of each specification parameter
        # and each changepoint parameter

        param_names = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            param_names += [f'{var}_{i}' for var in dmat.columns]

        # for each changepoint
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            param_names += [f'cp_{var}_{i}' for var in dmat.columns]

        # for each error structure (only 1 right now)
        param_names += ['sigma_0']

        return param_names


    def logp(self, point, data=None):
        if data is None:
            data = self.data
        return self.log_prior(point) + self.likelihood(point, data=data)


    def set_priors(self, priors=None):
        pass
        if priors is None:
            self.__set_data_informed_priors()
        else:
            # TODO
            # the plan is to accept frozen scipy distributions
            # e.g., scipy.stats.norm(loc=10, scale=3.7)
            # the pdfs of these frozen distributions can then be used
            # to generate log priors at sampled locations during sampling
            pass


    def __set_data_informed_priors(self):
        # TODO
        # construct priors based on the data we have in self.data
        # intercepts ~ t(loc = 0, scale = 3 * sd(y), nu = 3)
        # slopes ~ t(loc = 0, scale = sd(y)/(max(x) − min(x)), nu = 3)
        # changepoints?
        # epsilon ~ gamma?

        y_sd = np.std(self.data[self.y_var])
        x_range = np.ptp(self.data[self.x_var])
        prior = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            for var in dmat.columns:
                if var == 'Intercept':
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = 3. * y_sd)]
                else:
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = y_sd/x_range)]

        # for now, only permit a series of "intercept only" changepoint
        # specifications, making the priors much more straightforward
        # the components of the Dirichlet will be summed in order
        # with each changepoint's location expressed as a proportion of the
        # observed range (i.e., min(x) = 0, max(x) = 1)
        # so in a 3-segment model, with proposed changepoints at .3 and .7,
        # would be conceptualized as an observation of [.3, .4, .3]
        # because the sums of the Dirichlet components 1 through k are:
        # .3 and .3 + .4 = .7.  The final sum (e.g., .3+.4+.3) is always 1.0.
        # So the prior probability of this particular set of proposed
        # changepoints (i.e., .3 and .7) would be:
        # scipy.stats.dirichlet([alpha1, alpha2, alpha3]).pdf([.3, .4, .3])
        #
        # so we need num_changepoints + 1 alphas to build our prior for
        # *all* changepoints
        alphas = np.ones(len(self.changepoint_dmatrices)+1)
        prior += [scipy.stats.dirichlet(alphas)]

        # later we will implement priors for a wider variety of cp specs
        # and will need to sort out the priors for such specifications
        #for i,dmat in enumerate(self.changepoint_dmatrices):
        #    # for each predictor in specification

        # for each error structure (only 1 right now)
        prior += [scipy.stats.halfnorm(loc = 0, scale=y_sd)]
        self.prior = prior


    def log_prior(self, point, debug=False):
        # this mostly works but the multivariate Dirichlet breaks it
        #logp = 0
        #for val, dist in zip(point, self.prior):
        #    print([val, dist])
        #    logp += dist.pdf(val)
        # or
        #logps = [dist.pdf(val) for d in self.prior for val in point]
        #return sum(logps)


        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        param_idx = 0
        logp = 0
        # for each segment
        for i in range(num_sgmt_params):
            logp += self.prior[param_idx].pdf(point[param_idx])
            param_idx += 1

        # for each changepoint
        # this currently assumes a set of *non-parametric* changepoints
        # with a Dirichlet prior, so check to make sure we're not getting
        # something that conflicts with this
        if num_cp_params != len(self.changepoint_dmatrices):
            raise ValueError(
                "Received invalid changepoints specifications.  Only non-parametric changepoints are currenty implemented."
            )
        cps = point[param_idx:param_idx+num_cp_params]
        # the "x" values of the Dirichlet must sum to 1
        # the first N-1 are the changepoints
        # make sure they don't already sum to > 1
        # nor are any are < 0 or > 1
        if (sum(cps) > 0) or (min(cps) < 0) or (max(cps) > 1):
            return -np.inf
        # insert the last "x" for the Dirchlet
        cps += [1 - sum(cps)]
        logp += self.prior[param_idx].pdf(cps)

        # we should only have the error term left now
        assert(param_idx + 1 == len(self.prior))
        assert(param_idx + num_cp_params == len(point))

        # this should be the SD of the error term
        logp += self.prior[-1].pdf(point[-1])

        return logp


    def likelihood(self, point, data=None, debug=False):

        # predict outcome variable
        y_hat = self.predict(point, data=data, debug=debug)

        # likelihood of each observation
        y_dmat = self.outcome_dmatrix.to_numpy().reshape(-1)
        # the SD of the error term
        error_sd = point[-1]
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


    def predict(self, params, data=None, debug=False):

        params = list(params)
        if debug:
            print('params: ' + str(params))

        # the changepoints are defined relative to the min/max of the observed
        # x values in the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # so we need to warn users
        if (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: model was previously fit, but received data. Using previous observation window (e.g., min(x) and min(x)) to predict().', RuntimeWarning, stacklevel=2)

        # for each segment
        segment_params = []
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            segment_params += [params.pop(0) for var in dmat.columns]

        # for each changepoint
        cp_params = []
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            cp_params += [params.pop(0) for var in dmat.columns]

        # this param is not needed here, but should empty the list
        # which permits the subsequent validation
        #
        # for each error structure (only 1 right now)
        error_params = params.pop(0)

        # the parameter list should now be empty
        # if not, something has gone wrong
        if len(params) > 0:
            exp_num_params = sum(len(x) for x in segments_params)+ (len(cp_params)-1) + 1
            raise ValueError(
                "Received invalid initial parameter value guesses.  Expected " +str(exp_num_params) +" values, received " + str(exp_num_params+len(params)) +"."
            )

        # extract primary predictor variable from new data
        print('pname: ' + str(self.x_var))
        print('data: ' + str(data))
        x = data[self.x_var].to_numpy(dtype=float).reshape([-1,1])
        # reconstruct intercept vector
        intercept = np.ones_like(x) * self.segment_dmatrices[0]['Intercept'].to_numpy()[0]
        y_hat = np.zeros((x.shape[0],))
        zeros = np.zeros_like(x)

        # add in a dummy changepoint at min(x)
        cp_params = [0] + cp_params

        # untransform changepoints from [0,1] to [min(x), max(x)]
        # but make sure to use the original data's x values to do so
        cps = np.min(self.data[self.x_var].values) + (np.array(cp_params) * np.ptp(self.data[self.x_var].values))

        # for each segment
        print('num segments: ' +str(self.num_segments))
        for segment in range(self.num_segments):

            if debug:
                print('cp_params: ' +str(cp_params))
                print(self.changepoint_specifications[segment])
                print(data)
                print('cp: ' + str(cps))
                print(cps.shape)

            # transform predictor variable so that it "begins" at
            # the left edge of the segment
            #effective_x = x - cps[segment]
            #effective_x = np.clip(effective_x, 0, None)
            # make x values left of segment 0
            effective_x = x
            effective_x[effective_x < cps[segment]] = 0

            # insert intercept if specified
            print('in segment #' +str(segment))
            if 'Intercept' in self.segment_dmatrices[segment].columns:
                print('intercept shape: ' +str(intercept.shape))
                print('effective_x shape: ' +str(effective_x.shape))
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
                a = data[self.x_var]
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
                e = np.sum(segment_params[segment] * effective_x, axis=1).reshape([-1,1])

            # boolean indicating which datapoints are "in" current segment
            comp = data[self.x_var] < np.squeeze(cps[segment])
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





class bayes_old(bayes_base):

    def __init__(self, segments, y_var=None, x_var=None, changepoints=None, priors=None, data=None):

        super().__init__()

        # data must be set before models/changepoints
        self.set_data(data, validate=False)
        self.set_segments(segments, y_var=y_var, x_var=x_var, validate=False)
        self.set_changepoints(changepoints, validate=False)
        self.set_priors(priors=priors)
        self.validate_parameters()


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
        if num_cores is not None:
            poolholder = Pool(processes=num_cores)
        else:
            poolholder = nullcontext()
        with poolholder as pool:
            # generate sampler
            print('number of priors is '+str(len(self.prior)))
            sampler = emcee.EnsembleSampler(num_walkers, len(p0.shape[0]), self.logp,
                                            moves=emcee.moves.DESnookerMove(),
                                            pool=pool)
            if num_burnin > 0:
                p0 = sampler.run_mcmc(p0, num_burnin, progress=True)
                sampler.reset()
            sampler.run_mcmc(p0, nsamples, progress=True)

        # define variable names, it cannot be inferred from emcee
        var_names = self.gather_parameter_names()
        return arviz.from_emcee(sampler, var_names=var_names)


    def gen_start_point(self, num_walkers):

        # set up the starting point for each parameter and walker
        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        # with the Dirichlet-prior, we need num_changepoints + 1 alphas
        # so add 1 to this running sum
        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        # plus 1 because of the SD of the error term
        # at some point we need to have a list of error specifications
        num_dims = num_sgmt_params + num_cp_params + 1
        # initial coefficients for segment predictors
        # TODO: these should probably be informed by/sampled from the prior
        p0 = [0 for i in range(num_sgmt_params)]
        # initial coefficients for changepoint predictors
        # intercept set to mean of x_var
        p0 += [np.mean(self.data[self.x_var].to_numpy())]
        p0 += [0 for i in range(num_cp_params-1)]
        # assume a single
        #p0 += [1 for i in range(num_cp_params-1)]
        # generate a "fuzzed" version of those initial guesses
        # for each walker that we will use for MCMC
        p0 = np.array(p0)
        print((num_walkers, len(p0)))
        jitter = .04
        p0 = p0 + np.random.normal(scale=jitter, size=(num_walkers, len(p0)))
        # intial guess for error term SD must be positive
        # we use the sample SD and then scale it by some factor
        sdfactor = 0.1
        p0 = np.hstack([p0, sdfactor * np.std(self.data[self.y_var]) * np.random.random(num_walkers).reshape(-1,1)])
        print('gen_start_point believes we have ' +str(p0.shape)+' parameters')
        print(str(num_sgmt_params)+' for the segments')
        print(str(num_cp_params)+' for the changepoints')
        print('1 for the error')
        print('num_dims='+str(num_dims))
        return p0



    def gather_parameter_names(self):
        # for now, just generate strings of each specification parameter
        # and each changepoint parameter

        param_names = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            param_names += [f'{var}_{i}' for var in dmat.columns]

        # for each changepoint
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            param_names += [f'cp_{var}_{i}' for var in dmat.columns]

        # for each error structure (only 1 right now)
        param_names += ['sigma_0']

        return param_names


    def logp(self, point):
        return self.log_prior(point) + self.likelihood(point)


    def set_priors(self, priors=None):
        pass
        if priors is None:
            self.__set_data_informed_priors()
        else:
            # TODO
            # the plan is to accept frozen scipy distributions
            # e.g., scipy.stats.norm(loc=10, scale=3.7)
            # the pdfs of these frozen distributions can then be used
            # to generate log priors at sampled locations during sampling
            pass


    def __set_data_informed_priors(self):
        # TODO
        # construct priors based on the data we have in self.data
        # intercepts ~ t(loc = 0, scale = 3 * sd(y), nu = 3)
        # slopes ~ t(loc = 0, scale = sd(y)/(max(x) − min(x)), nu = 3)
        # changepoints?
        # epsilon ~ gamma?

        y_sd = np.std(self.data[self.y_var])
        x_range = np.ptp(self.data[self.x_var])
        prior = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            for var in dmat.columns:
                if var == 'Intercept':
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = 3. * y_sd)]
                else:
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = y_sd/x_range)]

        # for now, only permit a series of "intercept only" changepoint
        # specifications, making the priors much more straightforward
        assert(len(self.changepoint_dmatrices) == sum([len(x.columns) for x in self.changepoint_dmatrices]))

        # the components of the Dirichlet will be summed in order
        # with each changepoint's location expressed as a proportion of the
        # observed range (i.e., min(x) = 0, max(x) = 1)
        # so in a 3-segment model, proposed changepoints at .3 and .7
        # would be conceptualized as [.3, .4, .3] because
        # the sums of the Dirichlet components 1 through k are:
        # .3 and .3 + .4 = .7.  The final sum (e.g., .3+.4+.3) is always 1.0
        # fixing the final entry in vector so that it is 1 - the sum of the
        # first n-1 entries.
        # Thus, he prior probability of this particular set of proposed
        # changepoints (i.e., .3 and .7) would be:
        # scipy.stats.dirichlet([alpha1, alpha2, alpha3]).pdf([.3, .4, .3])
        #
        # so we need num_changepoints + 1 alphas to build our prior for
        # *all* changepoints
        alphas = np.ones(len(self.changepoint_dmatrices)+1)
        prior += [scipy.stats.dirichlet(alphas)]

        # later we will implement priors for a wider variety of cp specs
        # and will need to sort out the priors for such specifications
        #for i,dmat in enumerate(self.changepoint_dmatrices):
        #    # for each predictor in specification

        # for each error structure (only 1 right now)
        prior += [scipy.stats.halfnorm(loc = 0, scale=y_sd)]
        self.prior = prior


    def log_prior(self, point, debug=False):
        # this mostly works but the multivariate Dirichlet breaks it
        #logp = 0
        #for val, dist in zip(point, self.prior):
        #    print([val, dist])
        #    logp += dist.pdf(val)
        # or
        #logps = [dist.pdf(val) for d in self.prior for val in point]
        #return sum(logps)


        logp = 0
        param_idx = 0

        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        # for each segment
        for i in range(num_sgmt_params):
            logp += self.prior[param_idx].pdf(point[param_idx])
            param_idx += 1

        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        # for each changepoint
        # this currently assumes a set of *non-parametric* changepoints
        # with a Dirichlet prior, so check to make sure we're not getting
        # something that conflicts with this
        if num_cp_params != len(self.changepoint_dmatrices):
            raise ValueError(
                "Received invalid changepoints specifications.  Only non-parametric changepoints are currenty implemented."
            )
        cps = point[param_idx:param_idx+num_cp_params]
        # the "x" values of the Dirichlet must sum to 1
        # the first N-1 are the changepoints
        # make sure they don't already sum to > 1
        # nor are any are < 0 or > 1
        if (sum(cps) > 0) or (min(cps) < 0) or (max(cps) > 1):
            return -np.inf
        # insert the last "x" for the Dirchlet
        cps += [1 - sum(cps)]
        logp += self.prior[param_idx].pdf(cps)

        # we should only have the error term left now
        assert(param_idx + 1 == len(self.prior))
        assert(param_idx + num_cp_params == len(point))

        # this should be the SD of the error term
        logp += self.prior[-1].pdf(point[-1])

        return logp


    def likelihood(self, point, debug=False):

        # predict outcome variable
        y_hat = self.predict(point, fitting=True, debug=debug)

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


    def predict(self, params, data=None, fitting=False, debug=False):

        params = list(params)
        if debug:
            print('params: ' + str(params))

        # the changepoints are defined relative to the min/max of the observed
        # x values in the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # so we need to warn users
        if fitting and (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: model was previously fit, but received data. Using previous observation window (e.g., min(x) and min(x)) to predict().', RuntimeWarning, stacklevel=2)

        # for each segment
        segment_params = []
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            segment_params += [param.pop(0) for var in dmat.columns]

        # for each changepoint
        cp_params = []
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            cp_params += [param.pop(0) for var in dmat.columns]

        # this param is not needed here, but should empty the list
        # which permits the subsequent validation
        #
        # for each error structure (only 1 right now)
        error_params += pop(0)

        # the parameter list should now be empty
        # if not, something has gone wrong
        if len(params) > 0:
            exp_num_params = sum(len(x) for x in segments_params)+ (len(cp_params)-1) + 1
            raise ValueError(
                "Received invalid initial parameter value guesses.  Expected " +str(exp_num_params) +" values, received " + str(exp_num_params+len(params)) +"."
            )

        # extract primary predictor variable
        #print('pname: ' + str(self.x_var))
        x = patsy.dmatrix(self.x_var, data, return_type='dataframe')[self.x_var].to_numpy(dtype=float).reshape([-1,1])
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
                a = data[self.x_var]
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
            comp = data[self.x_var] < np.squeeze(cp)
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

