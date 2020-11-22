import numpy as np
import pandas as pd
import patsy
import scipy.stats
import scipy.optimize


class simple:
    def __init__(self, data=None, par_y='y', par_x='x', par_z='z'):

        self.par_y = par_y
        self.par_x = par_x
        self.par_z = par_z
        self.set_data(data)

        self.result = None


    def set_data(self, data=None):

        if data is None:
            return

        assert isinstance(
            data, pd.DataFrame
        ), "Received an invalid data object.  Data must be a pandas dataframe."

        assert self.par_y in data.columns, "par_y not found in provided dataframe."
        assert self.par_x in data.columns, "par_x not found in provided dataframe."
        assert self.par_z in data.columns, "par_z not found in provided dataframe."

        self.data = data


    def fit(self, x0=None, bounds=None):

        if bounds is not None:
            self.result = scipy.optimize.differential_evolution(
                self.model_logp,
                bounds,
                args=(self.data,),
                popsize=100,
                mutation=(0.7, 1),
                recombination=0.5
            )
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # need to set the stepsize more intelligently
        # but that requires scaling the parameters
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        elif x0 is not None:
            self.result = scipy.optimize.basinhopping(
                self.model_logp,
                x0,
                minimizer_kwargs={'args':(self.data,)},
                niter=100,
            )
        else:
            raise ValueError(
                "fit() requires x0 or bounds"
            )

    def summary(self):
        print(self.result)

    def change_point(self, params, data, mean=False):
        intercept = params[3]
        slope = params[4]

        linear = intercept + (slope * data[self.par_z])
        linear = np.clip(linear, 1e-6, 100)
        change_point_mean = np.exp(linear)

        if mean:
            return change_point_mean
        else:
            # need to handle the possibility that the estimated
            # changepoint for a case ends up being < 0
            mask = change_point_mean > 0
            change_point = np.zeros_like(change_point_mean)
            change_point[mask] = scipy.stats.poisson.rvs(change_point_mean[mask])
            return change_point


    def model_mean(self, params, data):
        # params = [preIntercept, preSlope, slopeDiff, changeIntercept, changeSlope]

        intercept = params[0]
        slope = params[1]
        slopeDiff = params[2]
        change_point = self.change_point(params, data, mean=True)
        isPost = data[self.par_x] > change_point

        mu = (
            intercept
            + (slope * data[self.par_x])
            + (isPost * (slopeDiff * (data[self.par_x] - change_point)))
        )

        mu = np.clip(mu, 1e-6, 100)
        return np.exp(mu)


    def model_logp(self, params, data):

        with np.errstate(divide="ignore"):
            tol = 1e-6
            p = scipy.stats.poisson.pmf(data[self.par_y], self.model_mean(params, data))
            p = np.clip(p, tol, 1-tol)
            return -np.sum(
                np.log(p)
            )

