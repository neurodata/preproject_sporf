# SPORF Preprocessing

This repo holds code to preprocess your data in a sporf-like manner.
DISCLAMER:  This will increase the size of your data.

## To compile:

Run

First, make virtualenv and activate

```
cd sporf_transform
pip install sklearn
pip install -e .
```

## To Use:

Run 

```python
from sporf_transform import sporf_ternaryColumns

"""
Suppose X_{n \times p} is your data matrix.
p is the number of features in your data matrix.
d is the number of new features created from sparse-oblique projections.
lam is the parameter in a Pois(lam) + 1 that determins the sparsity.
"""

p = 12
d = 12**4
lam = 1
A = sporf_ternaryColumns(p,d,lam)

### Create the projected data
# Xt = X @ A
```


## [Example01](sporf_transform/docs/Example01.ipynb)
