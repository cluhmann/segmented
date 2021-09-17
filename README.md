# segmented

segmented is a Python toolbox for performing segmented regression, with an initial focus on parametric changepoints characterizations.  The package currently provides two classes, each serving different purposes and each providing different feature sets.  The `segmented.demo` class is for demonstration and replication purposes only (please [see here](#demo) for details).  The `segmented.segmented` class provides the core functionality.  This class currently permits connected, parametric segmented linear regression models with identity link functions.  Eventually, this class will permit an arbitrary number of segments/nodes, alternative link functions, etc.

## Installation

segmented is written for Python 3.7+.  The latest release of segmented can be installed from PyPI using pip:

```
pip install segmented
```

## Segmented Class

Here, we briefly outline the functionality of the `segmented.segmented` class.  This class is intended to be the workhorse class of the segmented package.  This class currently permits the construction of segmented linear regression models that are a) connected, b) parametric (or non-parametric), and c) have identity link functions.  Future releases will relax these requirements (see below).

Let's see an example.  In what follows, we assume the following initial imports.  These are, coincidentally, the packages required by segmented:

```python
import pandas as pd
import numpy as np
import scipy.stats

import segmented as sgmt
```

Let's take a look at how the class works.  We'll start with a segmented regression model with a **non-parametric** changepoint.


```python
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([0,0,0,0,0,0,1,2,3,4])
data = pd.DataFrame({'x':x, 'y':y})


# construct a model
model = sgmt.segmented(['y~1+x', '0+x'], data=data)

# estimate parameter values
model.fit([7])

# check out result
print(model.summary())

```

Let's look at what we have here.  We first generate some data and then construct our model by calling `sgmt.segmented()`.  We then estimate the parameters of our model by calling `fit()` and finally we inspect the results by calling `summary()`.  Let's take a closer look at the model specification step.

The first argument to `sgmt.segmented()` is a list of [patsy](https://github.com/pydata/patsy) strings, each providing a specification for a separate segment of our model.  This listof segment specifications is always the first argument.  The second, named argument is the data we are modeling and must come in the form of a [pandas dataframe](https://pandas.pydata.org/docs).  In this example we have provided 2-segment specification.

Every segment in our segmented regression model, **including the first**, consists of what we refer to as a _node_ at the beginning (left edge) of the segment and a line that spans the interval between that segment's node (i.e., to the left) and the node associated with the next segment (i.e., to the right).  The first segment begins at with a node at x=min(x) that we will call x=T1.  The second segment begins with a node that marks the point at which the two segments are connected.  We refer to this point as x=T2.

The first segment specification provides 2 important pieces of information about our model.  First, it instructs `segmented` to treat `data['y']` as the outcome variable.  Second, it indicates that an intercept-like term will be estimated (cf. `'y~0+x'`).  Specifically, we estimate an offset, B0, such that y = f(T1) = f(min(x)) = B0.

The second item in the specification list describes the second segment and is a bit different.  There are two details in particular that are worth highlighting.

First, we explicitly omit an intercept by including the `0` in `'~0+x'`.  This means that `segmented()` will construct *connected* segments.  If we had instead included an intercept in these segment definitions (e.g., `'~1+x'` or, more implicitly, `'~x'`), we would instead permit the model to construct a *disconnected* model in which the segments need not meet at each node (not yet implemented).

Second, we have omitted the outcome variable (i.e., y) from the left-hand side of the second segment's specification. This is because, unlike the first segment's specification, the second segment's specification **does not** describe the relationship between the outcome variable and the predictor (at least not in a straightforward sense).  Instead, the second segment's specification describes how we wish the *difference* between the first segment and the second segment to be modeled.  The segment specification we provided suggests that the change occurring at the that segment's node (i.e., to the left) can be described by a simple change in (linear) slope (as a function of `data['x']`).  Thus, for each segment, there will be a single slope (coefficient) estimated for each segment: B1 for the first segment and B2 for the second segment.  The first segment will work a bit like conventional linear regression: y = B0 + B1 (x - T1).  However, within the second segment, y = B0 + B1 (x - T1) + B2 (x - T2)+.  Here, the (x - T2)+ = x - T2 if x > T2 and 0 otherwise.

We then call `fit()` and pass it a list of guesses about where the changepoints might be.  Because we have specified a model with 2 segements, we provide a list consisting of a guess aboute the single changepoint (i.e., `[7]`).

Finally, we inspect the results by calling `summary()`.  This will provide a variety of details regarding the optimization procedure.


Now let's take a look at a segmented regression model with a **parametric** changepoint.


```python
# generate some data
x = np.array([0,1,2,3,4,5,6,7,8,9])
b = np.array([0,1,-1])
cp = [2,0]
y = b[0] + (b[1] * x)
y+= (b[2] * np.clip(x-cp[0], 0, None))
z = cp[1] * np.ones_like(x) # not used
data = pd.DataFrame({'x':x, 'y':y, 'z':z})

# construct a model
model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)

# estimate parameter values
model.fit([0, 0, 0, data['x'].median(), 0, 1])

# check out result
print(model.summary())

```

This example is very similar to the non-parametric example.  We generate some data and then construct our model by passing a list of specifications and our data.  However, we also pass in a list of changepoint specifications.  Because we specified a model with 2 segments, we supply a single changepoint specification.  Here, we have specified that the changepoint will be a linear function of the variable z.  We have included an intercept term (the `'1'` in `'1+z'`).  So the changepoint, T2, that sits "in between" the first and second segments will be located at T2 = G0 + G1*z.  When we call `fit()` we supply a list of preliminary parameter values.  There are 6 values in this list: B0, B1, B2, G0, G1, and the log of the standard deviation of the (normally distributed) error term.




### Current limits (to be relaxed in future releases):

**Two segments maximum**

Segmented currently only permits 2 segments (1 changepoint).  This will be relaxed in future releases, first for non-parametric changepoint model (already implemented) and subsequently for parametric changepoint models (where things can get a bit more complicated).

**Identity link functions**

Segmented currently only permits identity link functions with normally-distributed error structures.  Additional link functions (e.g., log-linear) and error functions (e.g., Student's t) will added in future releases.

**One predictor maximum**

Segmented currently only permits a single predictor variable in the segment specifications (e.g., x).  We plan to relax this constraint in a future release, but the API may need to be reivsed to allow users to indicate which, of many, predictor is the primary predictor (i.e., the dimension along which the changepoints are defined).

**Connected Models Only**

Segmented currently only permits connected models.  This means that the second segment's specification **may not** include an intercept-like term.  That is, this is valid:

```python
# valid
model = sgmt.segmented(['y~1+x', '0+x'], changepoints=['1+z'], data=data)
```

This is **not** valid:

```python
# not valid
model = sgmt.segmented(['y~1+x', '1+x'], changepoints=['1+z'], data=data)
```

Disconnected models will be permitted in a future release.



## <a name="demo"></a> Demo Class

This class embodies a segmented regression model with parametric node placement.  The specification of both the segmented regression model itself and the specification of the parametric node placement are log-linear with Poisson error structures.  The `segmented.demo` class also assumes a single node (i.e., two segments).

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
model = sgmt.demo(data=data)

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



