import pytest

import numpy as np
import pandas as pd
import segmented as sgmt

from pandas.testing import assert_frame_equal


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
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 5]))
    assert(model.coefs == pytest.approx([0,0,1]))

    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([2,2,2,2,2,2,3,4,5,6])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 5]))
    assert(model.coefs == pytest.approx([2,0,1]))

    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,1,2,3,4,5,7,9,11,13])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 5]))
    assert(model.coefs == pytest.approx([0,1,1]))

    # verify degenerate left-most node location
    x = np.array([10,11,12,13,14,15,16,17,18,19])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([14])
    assert(model.nodes == pytest.approx([x.min(), 15]))
    assert(model.coefs == pytest.approx([0,0,1]))


def test_connected_parametric_fit():
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    z = np.array([0,0,0,0,0,1,1,1,1,1])
    y = np.array([0,1,2,3,4,10,9,8,7,6])
    data = pd.DataFrame({'x':x, 'y':y, 'z':z})
    model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)
    model.fit([.1, .1, .1, .1, .1, .1])
    print(model.result)
    print(model.nodes)
    print(model.coefs)
    # assert something here



