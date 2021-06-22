import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize


class demo:
    """This class is used for illustrative and replication purposes.  This object embodies a segmented regression model with parametric node placement.  The  specification of both the segmented regression model itself and the specification of the parametric node placement are log-linear with Poisson error structures and assumes a single node (i.e., two segments):

    mu_Y = exp(B_0 + (B_1 * X) + (B_2 * (X - exp(g_0 + (g_1 * Z)))))

    This specification has five parameters: B_0, B_1, B_2, g_1, and g_2.  B_0 is the intercept, B_1 is the slope of the first segment, B_2 is the change in slope occuring at the node.  The placement of the node is parametric, with its specification including an intercept, g_0, and a slope, g_1, relating Z and node placement.

    Please note that this model, unlike Luhmann & Clouston, assumes that T_0, the degenerate node at the "left-most" end of the model, is zero.  The model used here, in which T_0 = 0, means that the intercept, B_0, will be defined relative to the orgin (as in conventional regression).  Luhmann & Clouston assume that T_0 = min(X), which meant that B_0 was instead "intercept-like" and insensitive to the distribution of X.  For this reason, the demo object should only be applied to data with strictly positive values of X.

    """
    def __init__(self, data=None, par_y='y', par_x='x', par_z='z'):
        """
        Args:
           data (dataframe): Data to be used for estimation.  Columns contain relevant variables (i.e., outcome, predictor, co-variable).

           par_y (str): Name of dataframe column corresponding to the variable outcome.

           par_x (str): Name of dataframe column corresponding to the primary predictor variable.

           par_z (str): Name of dataframe column corresponding to the co-variable that is (assumed to be) related to the parametric node placements.


        """
        self.par_y = par_y
        self.par_x = par_x
        self.par_z = par_z
        self.set_data(data)

        self.result = None


    def set_data(self, data=None):
        """A method to set the data object.

        Args:
           data (dataframe): Data to use for estimation.  Columns contain relevant variables.

        """
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
        """Estimate model parameters.  Note that this function may take quite some time to execute because it currently uses scipy's differential_evolution procedure.

        Args:
           x0 (list-like): This argument is currently ignored.  Initial guess. Array of real elements of size (5,).  The entries of this array correspond to: [B_0, B_1, B_2, g_0, g_1].  See documentation of :class:`demo` for further details.

           bounds (list-like): This argument is currently required.  Each component bounds should be a tuple of the form (min, max). The fit method uses these to constrain the search for likely parameter values.  The entries of tuples should correspond to: [(B_0_min, B_0_max), (B_1_min, B_1_max), (B_2_min, B_2_max), (g_0_min, g_0_max), (g_1_min, g_1_max)].

        """

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
        #elif x0 is not None:
        #    self.result = scipy.optimize.basinhopping(
        #        self.model_logp,
        #        x0,
        #        minimizer_kwargs={'args':(self.data,)},
        #        niter=100,
        #    )
        else:
            raise ValueError(
                #"fit() requires x0 or bounds"
                "fit() requires bounds"
            )


    def summary(self):
       """Summarize the optimization procedure.  Currently returns a scipy.optimize.OptimizeResult object.
       """
       return self.result


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

