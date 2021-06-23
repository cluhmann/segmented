## segmented

segmented is a Python toolbox for performing segmented regression, with an initial focus on parametrically characterizing the location of nodes (i.e., changepoints, knots, etc.).

The current state of the package is primarily for demonstration and replication purposes.  The primary functionality current resides in the `segmented.demo` class.  This class embodies a segmented regression model with parametric node placement.  The  specification of both the segmented regression model itself and the specification of the parametric node placement are log-linear with Poisson error structures.  The `segmented.demo` class also assumes a single node (i.e., two segments). Future versions of the package will relax these assumptions, permitting variable numbers of segments/nodes, alternative link functions, and more.

## Installation

segmented is written for Python 3.7+.  The latest release of segmented can be installed from PyPI using pip:

```
pip install segmented
```


## Example


In the following example, we assume the following initial imports.  These are, coincidentally, the packages required by the segmented:

```python
import pandas as pd
import numpy as np
import scipy.stats

import segmented
```

Let's generate some synthetic data to use for estimation.

```python
# number of datapoints in our data set
n_samples = 100

# sample x and z uniformly [0,100]
rng = np.random.default_rng()
x = 100 * rng.random(size=n_samples)
z = 100 * rng.random(size=n_samples)
data = pd.DataFrame({'x':x, 'z':z})

# define parameters
b = [3.5, -.015, .025]
g = [np.log(25), .011]

# generate y
node_mean = np.exp(g[0] + (g[1] * data['z']))
nodes = scipy.stats.poisson.rvs(node_mean)
y_mean = np.exp(
            b[0]
            + (b[1] * data['x'])
            + ((data['x'] > nodes) * (b[2] * (data['x'] - nodes)))
            )
data['y'] = scipy.stats.poisson.rvs(y_mean)
```

Here, we have generated data reflecting 2 segments separated by a single node.  This node is parametric: its location is dependent on `data['z']`.  Now that we have our data, let's construct and fit our model.

```python
# construct model
model = segmented.demo(data=data)

# fit parametric node placement segmented regression model
model.fit(bounds=[(-5,5), (-.05,.05), (-.05,.05), (-5,5), (-.05,.05)])
```

Here, we initialize the demonstration model by passing in our data to the initialization.  We then call `fit()` to begin the estimation/optimization process, providing bounds to constrain the parameter values considered.  Once this function call returns, we can unpack our results and see what happened.

```python
# print summary of optimization
print(model.summary())

# compare parameter estimates to true values
for i,j in zip(b+g, model.result.x):
    print(f'{i:0.4f}\t-\t{j:0.4f}')
```

Currently, `summary()` returns a `scipy.optimize.OptimizeResult` object.
