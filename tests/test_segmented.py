import pytest

import arviz as az
import numpy as np
import pandas as pd
import segmented as sgmt

from pandas.testing import assert_frame_equal

rng = np.random.default_rng(12345)

def test_init():

    # 2 segments with *nonparametric* changepoints
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([0,0,0,0,0,0,1,2,3,4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    # check segmented object
    assert(isinstance(model, sgmt.segmented))
    # check number of segments
    assert(isinstance(model.num_segments, int))
    assert(model.num_segments == 2)
    # check model data
    assert(isinstance(model.data, pd.DataFrame))
    assert_frame_equal(model.data, data, check_dtype=True)
    # check models
    assert(isinstance(model.segment_specifications, list))
    assert(len(model.segment_specifications) == 2)
    assert(model.segment_specifications == ['1+x', '0+x'])
    # check variables
    assert(isinstance(model.outcome_var_name, str))
    assert(model.outcome_var_name == 'y')


    # 2 segments with *parametric* changepoints
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    z = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,0,0,0,0,0,1,2,3,4])
    data = pd.DataFrame({'x':x, 'y':y, 'z':z})
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['z'], data=data)
    # check segmented object
    assert(isinstance(model, sgmt.segmented))
    # check number of segments
    assert(isinstance(model.num_segments, int))
    assert(model.num_segments == 2)
    # check model data
    assert(isinstance(model.data, pd.DataFrame))
    assert_frame_equal(model.data, data, check_dtype=True)
    # check models
    assert(isinstance(model.segment_specifications, list))
    assert(len(model.segment_specifications) == 2)
    assert(model.segment_specifications == ['1+x', '0+x'])
    # check variables
    assert(isinstance(model.outcome_var_name, str))
    assert(model.outcome_var_name == 'y')


    # passing bad data
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data=[])
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data=['nonsense'])
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data='nonsense')
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data=12345)

    # missing outcome variable
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        z = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['1+x', '1+z'], data=data)

    # outcome variable mentioned in second model specification
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['y~1+x', 'y~1+x'], data=data)

    # not yet implemented?
    # no intercept in the first model specification
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['y~0+x', 'y~0+x'], data=data)

    # asking for segments with different predictors
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        z = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['y~1+x', '1+z'], data=data)

    # not yet implemented
    # asking segments with multiple predictors
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        z = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['y~1+x+z', '1+z'], data=data)

    # not yet implemented
    # requesting more than 2 segments
    with pytest.raises(NotImplementedError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y})
        model = sgmt.segmented(['y~1+x', '0+x', '0+x'], data=data)

    # not yet implemented
    # specifying disconnected segments
    with pytest.raises(ValueError):
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y})
        model = sgmt.segmented(['y~1+x', '1+x'], data=data)


def test_connected_nonparametric_fit():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,0,0,0,0,0,1,2,3,4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.nonparametric(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.changepoint_coefs == pytest.approx([x.min(), 5]))
    assert(model.segment_coefs == pytest.approx([0,0,1]))

    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([2,2,2,2,2,2,3,4,5,6])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.nonparametric(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.changepoint_coefs == pytest.approx([x.min(), 5]))
    assert(model.segment_coefs == pytest.approx([2,0,1]))

    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,1,2,3,4,5,7,9,11,13])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.nonparametric(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.changepoint_coefs == pytest.approx([x.min(), 5]))
    assert(model.segment_coefs == pytest.approx([0,1,1]))

    # verify degenerate left-most node location
    x = np.array([10,11,12,13,14,15,16,17,18,19])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.nonparametric(['y~1+x', '0+x'], data=data)
    model.fit([14])
    assert(model.changepoint_coefs == pytest.approx([x.min(), 15]))
    assert(model.segment_coefs == pytest.approx([0,0,1]))



def test_connected_parametric_fit():

    data = pd.DataFrame({'x':[0,5], 'y':[0,0], 'z':[0,1]})
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)

    # test predictions

    ### these tests focus on the second (non-zero) data point
    ### the first is there to 'set' the left edge of the first segment
    ### thus, the consequences of interest are seen on yhat[1]

    # data point should be in segment #1
    yhat = model.predict(data, params=[0, 1, 1, 7, 0, 1])
    assert(np.all(yhat == [0,5]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[0, 0, 1, 0, 0, 1])
    assert(np.all(yhat == [0,5]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[0, 1, 1, 0, 0, 1])
    assert(np.all(yhat == [0,10]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[0, 1, 1, 0, 7, 1])
    assert(np.all(yhat == [0,5]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[0, 1, 1, 0, 2, 1])
    assert(np.all(yhat == [0,8]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[0, 1, 1, 1, 2, 1])
    assert(np.all(yhat == [0,7]))

    # data point should be in segment #2
    yhat = model.predict(data, params=[3, 1, 1, 1, 2, 1])
    assert(np.all(yhat == [3,10]))

    # more involved tests of predictions
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    # coefficients
    b = np.array([10, 1, -2])
    # non-parametric changepoint
    cp = 5
    # should be an inverted 'V'
    y = b[0] + (b[1] * x)
    y+= (b[2] * np.clip(x-cp, 0, None))

    # include covariable that is just a vector of ones
    data = pd.DataFrame({'x':x, 'y':y, 'z':np.zeros_like(x)})
    # have changepoint specification include intercept and covariable (z)
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)
    # inserting original parameter values should reproduce data
    yhat = model.predict(data, params=[b[0], b[1], b[2], cp, 0, 1])
    assert(np.all(yhat == y))

    # include covariable that is just a vector of whatever `cp` is
    data = pd.DataFrame({'x':x, 'y':y, 'z':cp*np.ones_like(x)})
    # have changepoint specification only include covariable (z) (no intercept)
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['0+z'], data=data)
    # original parameter values and a changepoint *slope* of 1 should reproduce data
    yhat = model.predict(data, params=[b[0], b[1], b[2], 1, 1])
    assert(np.all(yhat == y))

    # test fit
    # linearly spaced
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    b = np.array([0,1,-1])
    cp = [2,0]
    y = b[0] + (b[1] * x)
    y+= (b[2] * np.clip(x-cp[0], 0, None))
    z = cp[1] * np.ones_like(x) # not used
    data = pd.DataFrame({'x':x, 'y':y, 'z':z})
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)
    model.fit([0, 0, 0, data['x'].median(), 0, 1])
    assert( model.result.x[0:-1] == pytest.approx(np.hstack((b,cp)), abs=.05) )
    assert( model.result.x[-1] < -3 )




def test_bayes_nonparametric():

    data = pd.DataFrame({'x':[0,5], 'y':[0,0], 'z':[0,1]})

    # 2 connected segments, non-parametric changepoint
    model = sgmt.bayes_nonparametric(['y~1+x', '0+x'], data=data)

    # 3 connected segments, static changepoints
    model = sgmt.bayes_nonparametric(['y~1+x'] + 2*['0+x'], data=data)

    # 2 disconnected segments, static changepoint
    model = sgmt.bayes_nonparametric(['y~1+x', '1+x'], data=data)

    # 2 connected segments, covariates, and parametric changepoints
    # here, income is a covariate in the model so it is ambiguous which
    # varaible the changepoint will operate over
    # so we need to specify it explicitly
    data = pd.DataFrame({'score':[0,0], 'time':[0,5], 'income':[12,33], 'IQ':[0,1]})
    model = sgmt.bayes_nonparametric(['score~1+time+income', '1+time+income'],
                        x_var='time',
                        data=data)

    # focus on estimation
    data = pd.DataFrame({'y':rng.random(size=100), 'x':rng.random(size=100)})
    model = sgmt.bayes_nonparametric(['y~1+x', '0+x'], data=data)
    # fit model
    idata = model.fit(num_samples=50, num_burnin=0, num_walkers=10)
    # summarize the model estimation
    az.summary(idata)

    # 3 connected segments, static but explicit changepoints
    model = sgmt.bayes_nonparametric(['y~1+x'] + 2*['0+x'], changepoints=2*['1'], data=data)

    # 2 disconnected segments, static but explicit changepoint
    model = sgmt.bayes_nonparametric(['y~1+x', '1+x'], changepoints=['1'], data=data)


    # we also allow explicit indication of the the outcome
    # variable as it might ease the writing of segment specifications
    data = pd.DataFrame({'y':[0,0], 'x':[0,5]})
    sgmt.bayes_nonparametric(5 * ['1+x'], changepoints=4*['1'], y_var='y', data=data)

    # at some point we can permit these sort of implicit specs
    # fill in changepoints with intercepts
    #sgmt.bayes_nonparametric(['y', '1+x', '1+x'], data=data)
    # equivalent to:
    # sgmt.bayes_nonparametric(['y', '1+x', '1+x'], changepoints=['1', '1','1'], data=data)

    # focus on estimation
    data = pd.DataFrame({'y':rng.random(size=100), 'x':rng.random(size=100)})
    model = sgmt.bayes_nonparametric(['y~1+x', '0+x'], changepoints=['1'], data=data)
    # fit model
    idata = model.fit(num_samples=50, num_burnin=0, num_walkers=10)
    # summarize the model estimation
    az.summary(idata)


def test_bayes_parametric_simple():

    data = pd.DataFrame({'y':rng.random(size=100),
                         'x':rng.random(size=100),
                         'z':rng.random(size=100)})

    # 2 connected segments, parametric changepoint
    model = sgmt.bayes_parametric(['y~1+x', '0+x'], changepoints=['1+z'], data=data)

    # fit model
    idata = model.fit(num_samples=50, num_burnin=0, num_walkers=50)
    # summarize the model estimation
    az.summary(idata)


def test_bayes_parametric_disconnected():

    data = pd.DataFrame({'x':[0,5], 'y':[0,0], 'z':[0,1]})

    # 2 DISconnected segments, parametric changepoint
    model = sgmt.bayes_parametric(['y~1+x', '1+x'], changepoints=['1+z'], data=data)


def test_bayes_parametric_complex():

    # 2 connected segments, covariates, and parametric changepoints
    # here, income is a covariate in the model so it is ambiguous which
    # varaible the changepoint will operate over
    # so we need to specify it explicitly
    data = pd.DataFrame({'y':[0,0], 'x':[0,5], 'z1':[12,33], 'z2':[0,1]})
    model = sgmt.bayes_parametric(['y~1+x+z1', '0+x+z1'],
                            changepoints=['1+z2'],
                            x_var='x',
                            data=data)









