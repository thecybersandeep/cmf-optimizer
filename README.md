# cmf-optimizer

implementation of gradient-based CMF discovery for the [ramanujan machine project]

## the problem with naive optimization

the challenge looks simple  minimize `||M_X·M_Y(x+1) - M_Y·M_X(y+1)||`. but if you just parameterize M_X and M_Y as generic polynomial matrices and throw gradient descent at it, you get nowhere. the conservation constraint is highly non-convex and you land in trivial local minima every time.

<img width="1878" height="1532" alt="image" src="https://github.com/user-attachments/assets/6ca4f817-e12f-419b-a47d-f6793b0a4702" />


## what actually works

the trick is realizing that all known CMFs come from the f,f̄ construction - and more importantly, f̄ is never independent of f. looking at the known solutions:

| constant | f | f̄ | relationship |
|----------|---|-----|--------------|
| ln(2) | x+y | x-y | f̄ = f(x,-y) |
| ζ(2) | 2x²+2xy+y² | -2x²+2xy-y² | f̄ = -f(-x,y) |
| ζ(3) | x³+2x²y+2xy²+y³ | -x³+2x²y-2xy²+y³ | f̄ = f(-x,y) |

there's a hidden symmetry here. f̄ is always f under some involution (negating x, y, or both, possibly with a sign flip). this isn't mentioned anywhere in their papers but it's what makes the search tractable.

once you constrain f̄ = T(f) for some transformation T, you're only optimizing coefficients of f, and the linear + quadratic conditions become much easier to satisfy.

## the construction

```
M_X = [[0, b(x)], [1, a(x,y)]]
M_Y = [[f̄, b(x)], [1, f]]

where:
  a(x,y) = f - f̄(x+1,y)
  b(x) = f(x,0)·f̄(x,0) - f(0,0)·f̄(0,0)
```

f and f̄ must satisfy:
- **linear**: `f(x,y) - f̄(x+1,y) = f(x+1,y-1) - f̄(x,y-1)` (diagonal invariance)
- **quadratic**: `f·f̄` separates as `g(x) + h(y)` (no mixed terms)

## usage

```python
from cmf_ffbar_optimizer import search_integer_cmfs, gradient_optimize_cmf, verify_known_cmfs

# verify against known CMFs
verify_known_cmfs()

# integer coefficient search
solutions = search_integer_cmfs(degree=2, ansatz='y_neg', coef_range=3)

# gradient optimization
sol = gradient_optimize_cmf(degree=2, ansatz='neg_x_neg', n_trials=50)
```

## status

verified correct on all known CMFs. finds valid families. the algebraic structure that makes f·f̄ separate while satisfying the linear condition is very restrictive at higher degrees.

## deps

```
numpy scipy sympy
```
