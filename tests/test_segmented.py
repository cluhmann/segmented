import pytest

import numpy as np
import pandas as pd
import segmented as sgmt

from pandas.testing import assert_frame_equal


def test_init():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([0,0,0,0,0,0,1,2,3,4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    assert(isinstance(model, sgmt.segmented))
    assert(isinstance(model.num_segments, int))
    assert(model.num_segments == 2)
    assert(isinstance(model.data, pd.DataFrame))
    assert_frame_equal(model.data, data, check_dtype=True)
    assert(isinstance(model.models, list))
    assert(len(model.models) == 2)
    assert(model.models == ['1+x', '0+x'])
    assert(isinstance(model.outcome_var_name, str))
    assert(model.outcome_var_name == 'y')
    assert(isinstance(model.predictor_var_name, str))
    assert(model.predictor_var_name == 'x')

    # passing bad data
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data=[])
    with pytest.raises(ValueError):
        model = sgmt.segmented(['y~1+x', '0+x'], data=['nonsense'])

    # asking segments with different predictors
    with pytest.raises(ValueError):
        x = np.array([1,2,3,4,5,6,7,8,9,10])
        z = np.array([1,2,3,4,5,6,7,8,9,10])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y, 'z':z})
        model = sgmt.segmented(['y~1+x', '1+z'], data=data)

    # not yet implemented
    # specifying disconnected segments
    with pytest.raises(ValueError):
        x = np.array([1,2,3,4,5,6,7,8,9,10])
        y = np.array([0,0,0,0,0,0,1,2,3,4])
        data = pd.DataFrame({'x':x, 'y':y})
        model = sgmt.segmented(['y~1+x', '1+x'], data=data)


def test_connected_nonparametric_fit():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([0,0,0,0,0,0,1,2,3,4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 6]))
    assert(model.coefs == pytest.approx([0,0,1]))

    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([2,2,2,2,2,2,3,4,5,6])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 6]))
    assert(model.coefs == pytest.approx([2,0,1]))

    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([0,1,2,3,4,5,7,9,11,13])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([7])
    assert(model.nodes == pytest.approx([x.min(), 6]))
    assert(model.coefs == pytest.approx([0,1,1]))

    # test degenerate left-most node location
    x = np.array([10,11,12,13,14,15,16,17,18,19])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    data = pd.DataFrame({'x':x, 'y':y})
    model = sgmt.segmented(['y~1+x', '0+x'], data=data)
    model.fit([14])
    assert(model.nodes == pytest.approx([x.min(), 15]))
    assert(model.coefs == pytest.approx([0,0,1]))






