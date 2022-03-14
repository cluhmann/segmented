import warnings

import arviz as az
import emcee
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import patsy

from abc import ABC, abstractmethod
from contextlib import nullcontext
from multiprocessing import Pool

tol = 1e-6



class bayes_base(ABC):

    def __init__(self, segments, y_var=None, x_var=None, changepoints=None, priors=None, data=None):

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

        # data must be set before models/changepoints
        self.set_data(data, validate=False)
        self.set_segments(segments, y_var=y_var, x_var=x_var, validate=False)
        self.set_changepoints(changepoints, validate=False)
        self.set_priors(priors=priors)
        self.validate_parameters()


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


    def set_num_segments(self, num_segments, validate=True):

        self.num_segments = num_segments

        if validate:
            # validate
            self.validate_parameters()


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


    def set_priors(self, priors=None):
        pass
        if priors is None:
            self._set_data_informed_priors()
        else:
            # accept a list of frozen scipy distributions
            # e.g., scipy.stats.norm(loc=10, scale=3.7)
            # the pdfs of these frozen distributions can then be used
            # to generate log priors at sampled locations during sampling
            self.prior = priors


    @abstractmethod
    def _set_data_informed_priors(self):
        raise NotImplementedError("Must override _set_data_informed_priors()")


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
        if self.changepoint_specifications is not None:
            if not isinstance(self.changepoint_specifications, list):
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
        if isinstance(self.changepoint_specifications, list) and (self.num_segments is not None):
            if len(self.changepoint_specifications) != self.num_segments - 1:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the specified number of segments."
                )
        if isinstance(self.changepoint_specifications, list) and isinstance(self.segment_specifications, list):
            if len(self.changepoint_specifications) != len(self.segment_specifications)-1:
                raise ValueError(
                    "Number of segments implied by changepoint specification conflicts with the number implied y the model specification."
                )

        # if number of segments is implied but not set, set it
        if self.num_segments is None:
            self.set_num_segments(len(self.segment_specifications))


    def fit(self, num_samples=500, num_burnin=200, num_walkers=30, num_cores=None):

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

            for sample in sampler.sample(p0, iterations=num_samples, progress=True):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue
                print(
                    "Mean acceptance %: {0:.3f}\tmean autocorrelation: {1:.3f}".format(
                        np.mean(sampler.acceptance_fraction),
                        np.mean(sampler.get_autocorr_time(tol=0))
                    )
                )

            #sampler.run_mcmc(p0, num_samples, progress=True)
            

        print(
            "Mean acceptance fraction: {0:.3f}".format(
                np.mean(sampler.acceptance_fraction)
            )
        )

        # define variable names, it cannot be inferred from emcee
        var_names = self.gather_parameter_names()
        return az.from_emcee(sampler, var_names=var_names)


    def gen_start_point(self, num_walkers):
        if True:
            return self.gen_start_point_byprior(num_walkers)
        else:
            return self.gen_start_point_byhand(num_walkers)


    @abstractmethod
    def gen_start_point_byprior(self, num_walkers):
        raise NotImplementedError("Must override gen_start_point_byprior()")


    @abstractmethod
    def gen_start_point_byprior(self, num_walkers):
        raise NotImplementedError("Must override gen_start_point_byhand()")


    def logp(self, point, data=None):
        if data is None:
            data = self.data
        log_prior = self.log_prior(point)
        if log_prior == -np.inf:
            #print('impossible log prior')
            return -np.inf
        else:
            log_like = self.log_likelihood(point, data=data)
            #print('log prior: ' + str(log_prior) + ' log like:' + str(log_like))
            return log_prior + log_like


    @abstractmethod
    def log_prior(self, num_walkers):
        raise NotImplementedError("Must override log_prior()")


    def log_likelihood(self, point, data=None, debug=False):

        # predict outcome variable
        y_hat = self.predict(point, data=data, debug=debug)

        # likelihood of each observation
        y_dmat = self.outcome_dmatrix.to_numpy().reshape(-1)
        # the SD of the error term
        error_sd = point[-1]
        logp = scipy.stats.norm.logpdf(y_dmat, y_hat, error_sd)

        if debug:
            print('fitting')
            print('params: '+str(point))
            print('y_dmat: '+str(y_dmat.reshape([-1,1])))
            print('y_hat: '+str(y_hat))
            print('logp: '+str(logp))
            print('sumlogp: '+str(np.sum(logp)))

        #print('point='+str(point))
        #print('log likelihood='+str(np.sum(logp)))

        # negative log likelihood of entire data set
        return np.sum(logp)


    @abstractmethod
    def predict(point, data=None, debug=False):
        raise NotImplementedError("Must override predict()")


class bayes_nonparametric(bayes_base):

    def __init__(self, *args, **kwargs):
        super(bayes_nonparametric, self).__init__(*args, **kwargs)


    def set_changepoints(self, changepoints=None, validate=True):
        # if the changepoints are implicit, specify them explicitly here
        # as an "intercept-only" specification
        if changepoints is None:
            changepoints = ['1'] * (self.num_segments-1)

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
                    if (len(dmat.columns) > 1) or not ('Intercept' in dmat.columns):
                        raise ValueError(
                            "Changepoint specifications imply parametric changepoints. Please use bayes_parametric class instead."
                        )
                    self.changepoint_dmatrices += [dmat]

            else:
                raise ValueError(
                    "Changepoints must be patsy strings."
                )

        if validate:
            # validate
            self.validate_parameters()


    def gen_start_point_byprior(self, num_walkers):
        # set up the starting point for each parameter and walker
        param_idx = 0
        p0 = []

        # for each segment
        for dmat in self.segment_dmatrices:
            # for each predictor in specification
            for var in dmat.columns:
                #print('p0 len:\t' + str(len(p0)))
                #print('new prior shape:\t' + str(self.prior[param_idx].rvs(size=num_walkers).shape))
                p0 += [self.prior[param_idx].rvs(size=num_walkers)]
                param_idx += 1

        # for each changepoint
        cps = self.prior[param_idx].rvs(size=num_walkers)
        cps = np.delete(cps, -1, axis=1)
        #print('p0 type:\t' + str(type(p0)))
        #print('p0 len:\t' + str(len(p0)))
        #print('CPS type:\t' + str(type(cps)))
        #print('CPS shape:\t' + str(cps.T.shape))
        p0 += [cps.ravel()]
        #p0 += cps.T
        param_idx += 1


        # for each error structure (only 1 right now)
        p0 += [self.prior[param_idx].rvs(size=num_walkers)]
        param_idx += 1
        #print('pidx:\t'+str(param_idx))
        #print('n priors:\t' + str(len(self.prior)))
        #print('n segments:\t' + str(self.num_segments))
        #print('p0 len:\t' + str(len(p0)))
        # number of priors + 1 
        assert(param_idx == len(self.prior))

        return np.array(p0).T


    def gen_start_point_byhand(self, num_walkers):

        # set up the starting point for each parameter and walker
        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        # with the Dirichlet-prior, we need num_changepoints + 1 alphas
        # so add 1 to this running sum
        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        # plus 1 because of the SD of the error term
        # at some point we need to have a list of error specifications
        num_dims = num_sgmt_params + num_cp_params + 1
        # initial coefficients for segment predictors
        p0 = [0 for i in range(num_sgmt_params)]
        # initial coefficients for changepoint predictors
        num_points = 2
        start = 1 / (num_points+1)
        stop = 1
        np.linspace(start, 1, num=num_points, endpoint=False)
        cp_guesses = np.linspace(1/(num_cp_params+1), 1, num=num_cp_params, endpoint=False)
        #print('num_cp_params: '+str(num_cp_params))
        #print('cp_guesses: '+str(cp_guesses))

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


    def _set_data_informed_priors(self):
        # construct priors based on the data we have in self.data
        # intercepts ~ t(loc = 0, scale = 3 * sd(y), nu = 3)
        # slopes ~ t(loc = 0, scale = sd(y)/(max(x) − min(x)), nu = 3)
        # changepoints ~ dirichlet([1] * (num_segments-1) )
        # epsilon ~ halfnormal(scale = sd(y))

        y_sd = np.std(self.data[self.y_var])
        x_range = np.ptp(self.data[self.x_var])
        #print('y_sd:\t' + str(y_sd))
        #print('x_range:\t' + str(x_range))
        #print('SLOPE PRIOR SD:\t' + str(y_sd/x_range))
        #assert(False)
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
        prior += [scipy.stats.halfnorm(loc = 0, scale=y_sd )]
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
        #print('logprior:\t' + str(logp))
        # for each segment
        for i in range(num_sgmt_params):
            logp += self.prior[param_idx].logpdf(point[param_idx])
            #print('point[param_idx]='+str(point[param_idx])+' type(prior)='+str(self.prior[param_idx].dist)+' logp:\t' + str(logp))
            #print('params:\t'+str(self.prior[param_idx].kwds))
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
        #print('cps: ' + str(cps))
        if (sum(cps) < 0) or (min(cps) < 0) or (max(cps) > 1):
            return -np.inf
        # insert the last "x" for the Dirchlet
        cps = np.hstack((cps, [1 - sum(cps)]))
        #print('cps: ' + str(cps))
        #print('logp of cps: ' + str(self.prior[param_idx].pdf(cps)))
        logp += self.prior[param_idx].logpdf(cps)

        #print('logprior:\t' + str(logp))

        # we should only have the error term left now
        #print('we are on param_idx ' + str(param_idx))
        #print('there are ' + str(len(self.prior)) + ' priors')
        assert(param_idx + 1 == len(self.prior)-1)
        #print('num_cp_params ' + str(num_cp_params))
        #print('len(point) ' + str(len(point)))
        assert(param_idx + 1 + num_cp_params == len(point))

        # this should be the SD of the error term
        logp += self.prior[-1].logpdf(point[-1])

        #print('logprior:\t' + str(logp))

        return logp


    def predict(self, params=None, data=None, debug=False):

        if params is None:
            raise ValueError(
                "Cannot predict without parameters being specified."
            )

        if data is None:
            data = self.data

        params = list(params)
        if debug:
            print('params: ' + str(params))

        # the changepoints are defined relative to the min/max of the observed
        # x values in the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # so we need to warn users
        if (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: received data that is different than model constructed with.  Using previous observation window (e.g., min(x) and min(x)) to predict().', RuntimeWarning, stacklevel=2)

        # for each segment
        # create a numpy array of parameters
        segment_params = []
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            segment_params += [np.array([params.pop(0) for var in dmat.columns])]

        # for each changepoint
        # create a numpy array of parameters
        cp_params = []
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            cp_params += [np.array([params.pop(0) for var in dmat.columns])]

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

        if debug:
            print('segment params:\t' + str(segment_params))
            print('cp params:\t' + str(cp_params))


        # extract primary predictor variable from new data
        #print('pname: ' + str(self.x_var))
        #print('data: ' + str(data))
        x = data[self.x_var].to_numpy(dtype=float).reshape([-1,1])
        # reconstruct intercept vector
        intercept = np.ones_like(x) * self.segment_dmatrices[0]['Intercept'].to_numpy()[0]
        y_hat = np.zeros((x.shape[0],))
        zeros = np.zeros_like(x)

        # add in a dummy changepoint at min(x)
        cp_params = np.append([0], cp_params)

        # untransform changepoints from [0,1] to [min(x), max(x)]
        # but make sure to use the original data's x values to do so
        cps = np.min(self.data[self.x_var].values) + (cp_params * np.ptp(self.data[self.x_var].values))

        # for each segment
        for segment in range(self.num_segments):

            if debug:
                print('cp_params: ' +str(cp_params))
                #print(self.changepoint_specifications[segment])
                #print(data)
                print('cp: ' + str(cps))
                print(cps.shape)

            # transform predictor variable so that it "begins" at
            # the left edge of the segment
            #effective_x = x - cps[segment]
            #effective_x = np.clip(effective_x, 0, None)
            # make x values left of segment 0
            obs_in_segment = x >= cps[segment]
            effective_x = x - cps[segment]
            effective_x[np.logical_not(obs_in_segment)] = 0
            effective_x_dmat = self.segment_dmatrices[segment].assign(**{self.x_var:effective_x})

            # add predictions to "running" totals
            y_hat += np.sum(segment_params[segment] * effective_x_dmat.values, axis=1)

            if debug:
                print('predicting segment #' +str(segment))
                print('segment params: ' +str(segment_params[segment]))
                print('cp_params: ' +str(cp_params[segment]))
                print('cp: ' +str(cps[segment]))
                # broken down for debugging
                a = data[self.x_var]
                aprime = cp_params[segment]
                b = segment_params[segment]
                c = effective_x
                print('xs: ' +str(x))
                print('eff. xs: ' +str(c))
                d = b * c
                print('product: ' +str(d))
                f = np.sum(d, axis=1)
                print('sum: ' +str(f))
                e = f.reshape([-1,1])
                print('reshape: ' +str(e))

            # boolean indicating which datapoints are "in" current segment
            #comp = data[self.x_var] < np.squeeze(cps[segment])
            # mask segment predictions to only appropriate datapoints
            #temp = np.where(comp, np.squeeze(zeros), np.squeeze(e))

            if debug:
                print('eshp'+str(e.shape))
                print('yhtshp'+str(y_hat.shape))
                print('cpshp'+str(cps.shape))
                print('cp: '+str(cps))
                print('cmpshp'+str(comp.shape))
                print('cmp: '+str(comp))
                print('obs_in_segment: '+str(obs_in_segment))
                print('temp: '+str(temp))
                print('e: ' +str(e))
                print('y_hatpreupdate: ' +str(y_hat))
                print('y_hatpostupdate: ' +str(y_hat+temp))


        return y_hat


class bayes_parametric(bayes_base):


    def __init__(self, *args, **kwargs):
        super(bayes_parametric, self).__init__(*args, **kwargs)


    def set_changepoints(self, changepoints=None, validate=True):

        # if the changepoints are implicit, specify them explicitly here
        # as an "intercept-only" specification
        if changepoints is None:
            raise ValueError(
                "Omitted changepoint specifications imply simple, nonparametric changepoints.  Please use bayes_nonparametric class instead."
            )

        # if we have received a list
        if isinstance(changepoints, list):

            if len(changepoints) > 1:
                    raise NotImplementedError(
                        "Parameteric changepoint models currently support a maximum of two segments (one changepoint)."
                    )

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

        # make sure specifications imply parametric changepoints
        num_nonparametric_cp_specs = 0
        for dmat in self.changepoint_dmatrices:
            if (len(dmat.columns) == 1) and ('Intercept' in dmat.columns):
                num_nonparametric_cp_specs += 1
        if num_nonparametric_cp_specs == len(self.changepoint_dmatrices):
            raise ValueError(
                "Changepoint specifications imply simple, nonparametric changepoints.  Please use bayes_nonparametric class instead."
            )

        if validate:
            # validate
            self.validate_parameters()


    def gen_start_point_byprior(self, num_walkers):
        # set up the starting point for each parameter and walker
        param_idx = 0
        p0 = []

        # for each segment
        for dmat in self.segment_dmatrices:
            print('segment')
            # for each predictor in specification
            for var in dmat.columns:
                #print('p0 len:\t' + str(len(p0)))
                #print('new prior shape:\t' + str(self.prior[param_idx].rvs(size=num_walkers).shape))
                p0 += [self.prior[param_idx].rvs(size=num_walkers)]
                param_idx += 1

        # for each changepoint
        for dmat in self.changepoint_dmatrices:
            print('changepoint')
            # for each predictor in specification
            for var in dmat.columns:
                #print('p0 len:\t' + str(len(p0)))
                #print('new prior shape:\t' + str(self.prior[param_idx].rvs(size=num_walkers).shape))
                cps = self.prior[param_idx].rvs(size=num_walkers)
                p0 += [cps.ravel()]
                #p0 += cps.T
                param_idx += 1

        #print('p0 type:\t' + str(type(p0)))
        #print('p0 len:\t' + str(len(p0)))
        #print('CPS type:\t' + str(type(cps)))
        #print('CPS shape:\t' + str(cps.T.shape))



        # for each error structure (only 1 right now)
        p0 += [self.prior[param_idx].rvs(size=num_walkers)]
        param_idx += 1
        #print('pidx:\t'+str(param_idx))
        #print('n priors:\t' + str(len(self.prior)))
        #print('n segments:\t' + str(self.num_segments))
        #print('p0 len:\t' + str(len(p0)))
        # number of priors + 1 
        assert(param_idx == len(self.prior))

        return np.array(p0).T


    def gen_start_point_byhand(self, num_walkers):

        # set up the starting point for each parameter and walker
        num_sgmt_params = sum([len(x.columns) for x in self.segment_dmatrices])
        # with the Dirichlet-prior, we need num_changepoints + 1 alphas
        # so add 1 to this running sum
        num_cp_params = sum([len(x.columns) for x in self.changepoint_dmatrices])
        # plus 1 because of the SD of the error term
        # at some point we need to have a list of error specifications
        num_dims = num_sgmt_params + num_cp_params + 1
        # initial coefficients for segment predictors
        p0 = [0 for i in range(num_sgmt_params)]
        # initial coefficients for changepoint predictors
        num_points = 2
        start = 1 / (num_points+1)
        stop = 1
        np.linspace(start, 1, num=num_points, endpoint=False)
        cp_guesses = np.linspace(1/(num_cp_params+1), 1, num=num_cp_params, endpoint=False)
        #print('num_cp_params: '+str(num_cp_params))
        #print('cp_guesses: '+str(cp_guesses))

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


    def _set_data_informed_priors(self):
        # construct priors based on the data we have in self.data
        # intercepts ~ t(loc = 0, scale = 3 * sd(y), nu = 3)
        # slopes ~ t(loc = 0, scale = sd(y)/(max(x) − min(x)), nu = 3)
        # changepoints ~ dirichlet([1] * (num_segments-1) )
        # epsilon ~ halfnormal(scale = sd(y))

        y_sd = np.std(self.data[self.y_var])
        x_sd = np.std(self.data[self.x_var])
        x_range = np.ptp(self.data[self.x_var])
        x_mean = np.mean(self.data[self.x_var])
        x_min = np.min(self.data[self.x_var])
        x_max = np.max(self.data[self.x_var])
        #print('y_sd:\t' + str(y_sd))
        #print('x_range:\t' + str(x_range))
        #print('SLOPE PRIOR SD:\t' + str(y_sd/x_range))
        #assert(False)
        prior = []
        # for each segment
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            for var in dmat.columns:
                if var == 'Intercept':
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = 3. * y_sd)]
                else:
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = y_sd/x_range)]

        # for each changepoint
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            for var in dmat.columns:
                if var == 'Intercept':
                    #prior += [scipy.stats.t(df = 3., loc = x_mean, scale = 3. * x_sd)]
                    #TODO
                    # this prior is too restrictive, should permit cps outside xrange
                    prior += [scipy.stats.uniform(loc = x_min, scale = x_max)]
                else:
                    #prior += [scipy.stats.t(df = 3., loc = 0., scale = x_sd/np.ptp(dmat[var]))]
                    prior += [scipy.stats.t(df = 3., loc = 0., scale = 1.)]

        # for each error structure (only 1 right now)
        prior += [scipy.stats.halfnorm(loc = 0, scale=y_sd )]
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
        #print('logprior:\t' + str(logp))
        # for each segment
        for _ in range(num_sgmt_params):
            logp += self.prior[param_idx].logpdf(point[param_idx])
            #print('point[param_idx]='+str(point[param_idx])+' type(prior)='+str(self.prior[param_idx].dist)+' logp:\t' + str(logp))
            #print('params:\t'+str(self.prior[param_idx].kwds))
            param_idx += 1


        # for each changepoint
        for _ in range(num_cp_params):
            logp += self.prior[param_idx].logpdf(point[param_idx])
            #print('point[param_idx]='+str(point[param_idx])+' type(prior)='+str(self.prior[param_idx].dist)+' logp:\t' + str(logp))
            #print('params:\t'+str(self.prior[param_idx].kwds))
            param_idx += 1

        #print('logprior:\t' + str(logp))

        # we should only have the error term left now
        #print('we are on param_idx ' + str(param_idx))
        #print('there are ' + str(len(self.prior)) + ' priors')
        assert(param_idx + 1 == len(self.prior))

        # this should be the SD of the error term
        logp += self.prior[-1].logpdf(point[-1])

        #print('logprior:\t' + str(logp))

        return logp


    def predict(self, params=None, data=None, debug=False):

        if params is None:
            raise ValueError(
                "Cannot predict without parameters being specified."
            )

        if data is None:
            data = self.data

        params = list(params)
        if debug:
            print('params: ' + str(params))

        # the changepoints are defined relative to the min/max of the observed
        # x values in the data **passed when object was created**
        # NOT on the data we are now using for prediction
        # so we need to warn users
        if (data is not None) and (not self.data.equals(data)):
            warnings.warn('WARNING: received data that is different than model constructed with.  Using previous observation window (e.g., min(x) and min(x)) to predict().', RuntimeWarning, stacklevel=2)

        # for each segment
        # create a numpy array of parameters
        segment_params = []
        for i,dmat in enumerate(self.segment_dmatrices):
            # for each predictor in specification
            segment_params += [np.array([params.pop(0) for var in dmat.columns])]

        # for each changepoint
        # create a numpy array of parameters
        cp_params = []
        for i,dmat in enumerate(self.changepoint_dmatrices):
            # for each predictor in specification
            cp_params += [np.array([params.pop(0) for var in dmat.columns])]

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

        if debug:
            print('segment params:\t' + str(segment_params))
            print('cp params:\t' + str(cp_params))


        # extract primary predictor variable from new data
        #print('pname: ' + str(self.x_var))
        #print('data: ' + str(data))
        x = data[self.x_var].to_numpy(dtype=float)#.reshape([-1,1])
        # reconstruct intercept vector
        intercept = np.ones_like(x) * self.segment_dmatrices[0]['Intercept'].to_numpy()[0]
        y_hat = np.zeros((x.shape[0],))
        zeros = np.zeros_like(x)

        # deal with changepoint
        # there should only be one (ensured during initial model specification)
        # but assert here so that we can refactor this once we allow for > 1 cp
        assert(len(self.changepoint_dmatrices) == 1)
        cp_dmat = self.changepoint_dmatrices[0]
        # create a list of changepoints
        # first is the left-most edge of the x-values
        cps = np.vstack((np.min(x) * np.reshape(np.ones_like(x), (1,-1)),
                        np.sum(cp_params[0] * cp_dmat.to_numpy(), axis=1)
                       ))

        # for each segment
        for segment in range(self.num_segments):

            if debug:
                print('cp_params: ' +str(cp_params))
                #print(self.changepoint_specifications[segment])
                #print(data)
                print('cp: ' + str(cps))
                print(cps.shape)

            # transform predictor variable so that it "begins" at
            # the left edge of the segment
            #effective_x = x - cps[segment]
            #effective_x = np.clip(effective_x, 0, None)
            # make x values left of segment 0
            #print('x: ' + str(x))
            #print('cps[segment]: ' + str(cps[segment]))
            obs_in_segment = x >= cps[segment]
            #print('obs_in_segment: ' + str(obs_in_segment))
            effective_x = x - cps[segment]
            #print('effective_x: ' + str(effective_x))
            effective_x[np.logical_not(obs_in_segment)] = 0
            #print('effective_x: ' + str(effective_x))
            # TODO
            # this is using the data in self.segment_dmatrices
            # should use data argument to generate a new dmatrix
            effective_x_dmat = self.segment_dmatrices[segment].assign(**{self.x_var:effective_x})
            #print('effective_x_dmat: ' + str(effective_x_dmat))

            # add predictions to "running" totals
            y_hat += np.sum(segment_params[segment] * effective_x_dmat.to_numpy(), axis=1)


            if False and segment == 1:
                print('DOrF')
                print(cps)
                print(cps[segment])
                print(cp_dmat.head())
                print(np.sum(cp_params[0] * cp_dmat.to_numpy(), axis=1))
                print(obs_in_segment)
                print(data[self.x_var])
                print(effective_x_dmat)
                assert(False)


            if False:
                print('predicting segment #' +str(segment))
                print('segment params: ' +str(segment_params[segment]))
                #print('cp_params: ' +str(cp_params[segment]))
                print('cp: ' +str(cps[segment]))
                # broken down for debugging
                a = data[self.x_var]
                aprime = cp_params[segment]
                b = segment_params[segment]
                c = effective_x
                print('xs: ' +str(x))
                print('eff. xs: ' +str(c))
                d = b * c
                print('product: ' +str(d))
                f = np.sum(d, axis=1)
                print('sum: ' +str(f))
                e = f.reshape([-1,1])
                print('reshape: ' +str(e))
                print('y_hat: ' +str(y_hat))

            # boolean indicating which datapoints are "in" current segment
            #comp = data[self.x_var] < np.squeeze(cps[segment])
            # mask segment predictions to only appropriate datapoints
            #temp = np.where(comp, np.squeeze(zeros), np.squeeze(e))

            if False:
                print('eshp'+str(e.shape))
                print('yhtshp'+str(y_hat.shape))
                print('cpshp'+str(cps.shape))
                print('cp: '+str(cps))
                #print('cmpshp'+str(comp.shape))
                #print('cmp: '+str(comp))
                print('obs_in_segment: '+str(obs_in_segment))
                print('temp: '+str(temp))
                print('e: ' +str(e))
                print('y_hatpreupdate: ' +str(y_hat))
                print('y_hatpostupdate: ' +str(y_hat+temp))

        return y_hat


