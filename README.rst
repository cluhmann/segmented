segmented is a Python toolbox for performing segmented regression.

************
Installation
************

segmented is written for Python 3.7+.  The latest release of segmented can be installed from PyPI using pip:

.. code-block::

    pip install segmented


*******
Roadmap
*******
* 0.0.1

  * two segments with a single, general changepoint

  * models are always connected

  * identity link functions

  * single predictor

* 0.0.2

  * two segments with a parametric changepoint

  * identity link functions

  * single predictor

* 0.0.3

  * permit unconnected models

  * two segments with a parametric changepoint

  * identity link functions

  * single predictor

* >= 0.0.4

  * user-specified number of segments (>2)

  * various link functions

  * etc.


*****
Usage
*****

.. code-block:: python

    import segmented as sgmt
    import pandas as pd

    data = pd.read_csv('mydata.csv')

    # construct a 2 segment model
    model = sgmt.segment(['y~1', '~0+x', '~0+x'], num_segments=2, data=data)

    # do inference
    model.fit()

    # check out result
    print(model.summary())

Let's look at what we have here.  We first read some data into a pandas dataframe.  We then construct our model by calling :code:`sgmt.segment()` and passing it a list of `patsy <https://github.com/pydata/patsy>`_ formulas.  We then use our data to estimate the parameters of our model.  Finally, we inspect the results.  Let's take a closer look at the model specification step.

Model Specification
*******************

The first argument to :code:`segment()` is a list of formulas that describe our model.  This is always the first argument.  The second, named argument is the data we are modeling.  The model we have defined here has two segments, so we specify 3 components.  We are likely to be most interested in the location of the node connecting the two segments, :math:`T_2`.  We also posit an additional node at :math:`x=min(x)` that we will call :math:`T_1`.

The first formula specifies an intercept-like term and provides 2 important pieces of information about our model.  First, it instructs :code:`segment` to treat :code:`data['y']` as our outcome variable.  Second, it indicates that an intercept-like term will be estimated (cf. :code:`'y~0'`).  Specifically, we will estimate an offset, :math:`\beta_0` such that :math:`y = f(T_1) = f(min(x)) = \beta_0`.

The next two elements in the definition list describe the two segments and are a bit different.  There are two details in particular that are worth highlighting.

First, we explicitly omit an intercept from these specifications by including the :code:`0` in :code:`'~0+x'`.  This means that :code:`segment()` will attempt to construct a series of *connected* segments (constraining each segment to meet adjacent segments at nodes).  If we had instead included an intercept in these segment definitions (e.g., :code:`'~1+x'` or, more implicitly, :code:`'~x'`), we would instead permit the model to construct a *disconnected* model in which the segments need not meet at each node.

Second, we have omitted the outcome variable (i.e., :code:`y`) from the left-hand side of the formula. This is because, unlike the intercept-like term, the segment definitions **do not** describe the relationship between the outcome variable and the predictor within that given segment (at least not in a straightforward sense).  Instead, the segment definition describes how we wish the *difference* between the current segment and the previous segment to be modeled.  Each of the segment definitions we have provided here suggests that the change occurring at the preceding node can be described by a simple change in (linear) slope (as a function of :code:`data['x']`).  Thus, for each segment, there will be a single slope (coefficient) estimated for each segment: :math:`\beta_1` for the first segment and :math:`\beta_2` for the second segment.  The first segment will work much like conventional linear regression: :math:`y=\beta_0+\beta_1 (x - T_1)`.  However, in the second segment, :math:`y=\beta_0 + \beta_1 (x - T_1) + \beta_2 (x - T_2)`.

Requiring a user to provide all the specifications may seem excessive, particularly when the user wishes to use the same specification for all segments.  However, it's easy enough to compactly specify a large number of segments.

.. code-block:: python

    # 3 segments, 2 changepoints
    model = sgmt.segment(['y~1+x', '0+x', '0+x'], data=data)

    # can we more compactly specify all the strings,
    # when we have identically specified segments? Yes!
    model = sgmt.segment(['y~1+x'] + 2 * ['0+x'], data=data)

    # so here would be how you specify a 10-segment model
    model = sgmt.segment(['y~1+x'] + 9 * ['0+x'], data=data)




Older API proposals:

.. code-block:: python

    model = sgmt.segment(['1', 'y~0+x', 'y~0+x'], data=data)

    # more compact specifications are also possible

    # specify 2 model specifications: pre- and post-changepoint
    model = sgmt.segment(['1', 'y~x', 'y~x'], data=data)

    # specify 2 model specifications: pre- and post-changepoint
    model = sgmt.segment('y~x', data=data)

    # specify a single model specification that will be used for each segment
    model = sgmt.segment(['y~x'], num_segments=2, data=data)

    # might need to provide some initial guesses at to-be-estimated parameters
    model = sgmt.segment(['y~1', 'y~x'], x0=.5, data=data)

    # might provide GLM capabilities
    model = sgmt.segment(['y~x'], family=['binomial'], num_segments=2, data=data)

    # estimate parameters
    model.fit()
    # check out estimates
    print(model.summary())


Another:

.. code-block:: python

    ###############################################################################
    # 2 segments, each segments has an intercept and slope associated with x

    # the changepoint is a single value of x
    model = sgmt.segment('y~x', changepoint='~1', num_segments=2, data=data)
    # is equivalent to:
    model = sgmt.segment('y~x', num_segments=2, data=data)

    # 2 segments, each segments has an intercept and slope associated with x
    # changepoint is parametric, with it's own intercept and slope associated with z
    model = sgmt.segment('y~x', changepoint='~z', num_segments=2, data=data)

    model = sgmt.segment(['y~x','y~x','y~x'], changepoint=['~z','~w'], data=data)


**********
References
**********
- TBD
