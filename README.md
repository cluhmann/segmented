# Segmented

segmented is a Python toolbox for segmented regression.

## Installation

segmented is written for Python 3.7+.  The latest release of segmented can be installed from PyPI using pip:

```pip install segmented```


## Roadmap
- 0.0.1
    - single change point
    - identity link functions
    - single predictor
- 0.0.2
    - single, parametric change point
    - identity link functions
    - single predictor
- >= 0.0.3
    - user-specified number of segments (>2)
    - various link functions
    - etc.


## How to Use segmented

```py
import segmented as sgmt
import pandas as pd

data = pd.read_csv('mydata.csv')

###############################################################################
# 2 segment model (1 change point)

# specify 2 model specifications: pre- and post-changepoint
model = sgmt.segment('y~x', data=data)

# specify 2 model specifications: pre- and post-changepoint
model = sgmt.segment(['y~x', 'y~x'], data=data)

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

```

OR

```py
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
```

# References
- TBD
