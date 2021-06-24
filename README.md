## segmented

segmented is a Python toolbox for performing segmented regression, with an initial focus on parametrically characterizing the location of nodes (i.e., changepoints, knots, etc.).

The package currently provides functionality for demonstration and replication purposes.  This functionality resides in the `segmented.demo` class.  This class embodies a segmented regression model with parametric node placement.  The  specification of both the segmented regression model itself and the specification of the parametric node placement are log-linear with Poisson error structures.  The `segmented.demo` class also assumes a single node (i.e., two segments).

Work has begun to relax these assumptions (permitting variable numbers of segments/nodes, alternative link functions, and more).  That functionality will reside in the `segmented.segmented` class.  The package currently permits connected, non-parametric segmented linear regression models with identity link functions.

## Installation

segmented is written for Python 3.7+.  The latest release of segmented can be installed from PyPI using pip:

```
pip install segmented
```


## Demo


In the following example, we assume the following initial imports.  These are, coincidentally, the packages required by segmented:

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


## Segmented class

Here is a quick example of the `segmented.segmented` class.

```python
import segmented as sgmt
import pandas as pd

data = pd.read_csv('mydata.csv')

# construct a 2 segment model
model = sgmt.segment(['y~1', '~0+x'], data=data)

# construct a 2 segment model
model = sgmt.segment(['y~1', '~0+x', '~0+x'], num_segments=2, data=data)

# do inference
model.fit()

# check out result
print(model.summary())

```

Let's look at what we have here.  We first read some data into a pandas dataframe.  We then construct our model by calling `sgmt.segment()` and passing it a list of `patsy <https://github.com/pydata/patsy>`_ formulas.  We then use our data to estimate the parameters of our model.  Finally, we inspect the results.  Let's take a closer look at the model specification step.

The first argument to `segmented.segmented()` is a list of formulas that describe our model.  This is always the first argument.  The second, named argument is the data we are modeling.  We have provided 2 segment specification and we are likely very interested in the location of the node connecting the two segments, T2.  We also posit an additional node at x=min(x) that we will call T1.

The first formula specifies an intercept-like term and provides 2 important pieces of information about our model.  First, it instructs `segment` to treat `data['y']` as our outcome variable.  Second, it indicates that an intercept-like term will be estimated (cf. `'y~0'`).  Specifically, we will estimate an offset, `\beta_0` such that `y = f(T_1) = f(min(x)) = \beta_0`.

The next two elements in the definition list describe the two segments and are a bit different.  There are two details in particular that are worth highlighting.

First, we explicitly omit an intercept from these specifications by including the `0` in `'~0+x'`.  This means that `segment()` will attempt to construct a series of *connected* segments (constraining each segment to meet adjacent segments at nodes).  If we had instead included an intercept in these segment definitions (e.g., `'~1+x'` or, more implicitly, `'~x'`), we would instead permit the model to construct a *disconnected* model in which the segments need not meet at each node.

Second, we have omitted the outcome variable (i.e., `y`) from the left-hand side of the formula. This is because, unlike the intercept-like term, the segment definitions **do not** describe the relationship between the outcome variable and the predictor within that given segment (at least not in a straightforward sense).  Instead, the segment definition describes how we wish the *difference* between the current segment and the previous segment to be modeled.  Each of the segment definitions we have provided here suggests that the change occurring at the preceding node can be described by a simple change in (linear) slope (as a function of `data['x']`).  Thus, for each segment, there will be a single slope (coefficient) estimated for each segment: beta1 for the first segment and beta2 for the second segment.  The first segment will work much like conventional linear regression: y=beta0+beta1 (x - T1).  However, in the second segment, y=beta0 + beta1 (x - T1) + beta2 (x - T2).

