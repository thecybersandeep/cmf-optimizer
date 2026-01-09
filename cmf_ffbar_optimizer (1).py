#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize
import sympy as sp
from sympy import symbols, expand, simplify, Poly, Matrix, Integer, Rational, nsimplify
from itertools import combinations, product as cartesian_product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

x, y = symbols('x y')


@dataclass
class CMFSolution:
    """A valid Conservative Matrix Field solution"""
    f: sp.Expr
    fbar: sp.Expr
    a: sp.Expr
    b: sp.Expr
    ansatz: str
    degree: int
    
    def __str__(self):
        return f"f={self.f}, fbar={self.fbar}"


def check_linear_condition(f: sp.Expr, fbar: sp.Expr) -> bool:
    """Check if f,fbar satisfy: f(x,y) - fbar(x+1,y) = f(x+1,y-1) - fbar(x,y-1)"""
    LHS = f - fbar.subs(x, x+1)
    RHS = f.subs([(x, x+1), (y, y-1)]) - fbar.subs(y, y-1)
    return simplify(expand(LHS - RHS)) == 0


def check_quadratic_condition(f: sp.Expr, fbar: sp.Expr) -> bool:
    """Check if f*fbar has no mixed x^i*y^j terms (i,j > 0)"""
    prod = expand(f * fbar)
    if prod == 0:
        return True
    poly = Poly(prod, x, y)
    for monom in poly.as_dict().keys():
        if monom[0] > 0 and monom[1] > 0:
            return False
    return True


def build_cmf_matrices(f: sp.Expr, fbar: sp.Expr) -> Tuple[sp.Matrix, sp.Matrix, sp.Expr, sp.Expr]:
    """Build CMF matrices M_X, M_Y from f, fbar"""
    a = expand(f - fbar.subs(x, x+1))
    b = expand(f.subs(y, 0) * fbar.subs(y, 0) - 
               f.subs([(x,0), (y,0)]) * fbar.subs([(x,0), (y,0)]))
    
    M_X = Matrix([[0, b], [1, a]])
    M_Y = Matrix([[fbar, b], [1, f]])
    
    return M_X, M_Y, a, b


def verify_conservation(M_X: sp.Matrix, M_Y: sp.Matrix) -> bool:
    """Verify M_Y * M_X(y+1) = M_X * M_Y(x+1)"""
    LHS = M_Y * M_X.subs(y, y+1)
    RHS = M_X * M_Y.subs(x, x+1)
    return simplify(expand(LHS - RHS)) == Matrix([[0,0],[0,0]])


def apply_ansatz(f: sp.Expr, ansatz: str) -> sp.Expr:
    """Generate fbar from f using transformation ansatz"""
    if ansatz == 'y_neg':      # fbar = f(x, -y)
        return f.subs(y, -y)
    elif ansatz == 'x_neg':    # fbar = f(-x, y)
        return f.subs(x, -x)
    elif ansatz == 'neg_x_neg': # fbar = -f(-x, y)
        return -f.subs(x, -x)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def search_integer_cmfs(degree: int, ansatz: str, coef_range: int = 3, 
                        max_nonzero: int = 5, verbose: bool = True) -> List[CMFSolution]:
    """
    Search for CMFs with small integer coefficients.
    
    Args:
        degree: Maximum polynomial degree
        ansatz: Transformation to generate fbar from f ('y_neg', 'x_neg', 'neg_x_neg')
        coef_range: Range of integer coefficients to try (-coef_range to +coef_range)
        max_nonzero: Maximum number of non-zero coefficients in f
        verbose: Print found solutions
    
    Returns:
        List of valid CMF solutions
    """
    monomials = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    n_coeffs = len(monomials)
    
    solutions = []
    seen = set()
    
    for n_nonzero in range(1, min(n_coeffs + 1, max_nonzero + 1)):
        for positions in combinations(range(n_coeffs), n_nonzero):
            for values in cartesian_product(range(-coef_range, coef_range + 1), repeat=n_nonzero):
                if all(v == 0 for v in values):
                    continue
                
                # Build f polynomial
                f = Integer(0)
                for pos, val in zip(positions, values):
                    i, j = monomials[pos]
                    f = f + val * x**i * y**j
                
                # Generate fbar using ansatz
                fbar = apply_ansatz(f, ansatz)
                
                # Check conditions
                if not check_linear_condition(f, fbar):
                    continue
                if not check_quadratic_condition(f, fbar):
                    continue
                
                # Build and verify CMF
                M_X, M_Y, a, b = build_cmf_matrices(f, fbar)
                if not verify_conservation(M_X, M_Y):
                    continue
                
                # Check if already found (up to scaling)
                f_str = str(expand(f))
                if f_str in seen:
                    continue
                seen.add(f_str)
                
                sol = CMFSolution(f=f, fbar=fbar, a=a, b=b, ansatz=ansatz, degree=degree)
                solutions.append(sol)
                
                if verbose:
                    print(f"  Found: {sol}")
    
    return solutions


def gradient_optimize_cmf(degree: int, ansatz: str, n_trials: int = 100,
                          verbose: bool = True) -> Optional[CMFSolution]:
    """
    Use gradient-based optimization to find CMFs.
    
    Minimizes: ||linear_condition||² + ||quadratic_condition||² + regularization
    
    Args:
        degree: Maximum polynomial degree for f
        ansatz: Transformation to generate fbar from f
        n_trials: Number of random restarts
        verbose: Print progress
    
    Returns:
        Best CMF solution found, or None if no valid solution
    """
    monomials = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    n_coeffs = len(monomials)
    
    def params_to_f(params):
        return sum(params[k] * x**monomials[k][0] * y**monomials[k][1] 
                   for k in range(n_coeffs))
    
    def compute_loss(params):
        f_expr = params_to_f(params)
        fbar_expr = apply_ansatz(f_expr, ansatz)
        
        # Linear condition error
        LHS = f_expr - fbar_expr.subs(x, x+1)
        RHS = f_expr.subs([(x, x+1), (y, y-1)]) - fbar_expr.subs(y, y-1)
        lin_diff = expand(LHS - RHS)
        
        lin_error = 0.0
        if lin_diff != 0:
            poly = Poly(lin_diff, x, y)
            for coef in poly.coeffs():
                lin_error += float(coef)**2
        
        # Quadratic condition error
        prod = expand(f_expr * fbar_expr)
        quad_error = 0.0
        if prod != 0:
            poly = Poly(prod, x, y)
            for monom, coef in poly.as_dict().items():
                if monom[0] > 0 and monom[1] > 0:
                    quad_error += float(coef)**2
        
        # Regularization to avoid trivial solution
        norm = np.sum(params**2)
        reg = 0.1 * (norm - 1)**2
        
        return lin_error + quad_error + reg
    
    best_params = None
    best_loss = float('inf')
    
    for trial in range(n_trials):
        # Random initialization with sparsity
        theta0 = np.random.randn(n_coeffs)
        if trial % 2 == 0:
            mask = np.random.random(n_coeffs) > 0.5
            theta0 *= mask
        
        if np.linalg.norm(theta0) > 1e-10:
            theta0 /= np.linalg.norm(theta0)
        
        result = minimize(compute_loss, theta0, method='L-BFGS-B', options={'maxiter': 1000})
        
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()
    
    if best_params is None or best_loss > 1e-6:
        return None
    
    f = params_to_f(best_params)
    fbar = apply_ansatz(f, ansatz)
    
    # Try to rationalize coefficients
    try:
        f = nsimplify(f, tolerance=1e-6, rational=True)
        fbar = nsimplify(fbar, tolerance=1e-6, rational=True)
    except:
        pass
    
    # Verify
    if not check_linear_condition(f, fbar) or not check_quadratic_condition(f, fbar):
        return None
    
    M_X, M_Y, a, b = build_cmf_matrices(f, fbar)
    if not verify_conservation(M_X, M_Y):
        return None
    
    return CMFSolution(f=f, fbar=fbar, a=a, b=b, ansatz=ansatz, degree=degree)


def find_all_cmfs(degree: int, method: str = 'integer', **kwargs) -> List[CMFSolution]:
    """
    Find CMFs at given degree using all ansatzes.
    
    Args:
        degree: Maximum polynomial degree
        method: 'integer' for brute-force, 'gradient' for optimization
        **kwargs: Additional arguments for search method
    
    Returns:
        List of all valid CMF solutions found
    """
    ansatzes = ['y_neg', 'x_neg', 'neg_x_neg']
    all_solutions = []
    
    for ansatz in ansatzes:
        print(f"  Trying ansatz: {ansatz}")
        
        if method == 'integer':
            solutions = search_integer_cmfs(degree, ansatz, **kwargs)
        else:
            sol = gradient_optimize_cmf(degree, ansatz, **kwargs)
            solutions = [sol] if sol else []
        
        all_solutions.extend(solutions)
    
    return all_solutions


def verify_known_cmfs():
    """Verify that known CMFs satisfy our implementation"""
    print("Verifying known CMFs:")
    
    known = [
        ("ln(2)", x+y, x-y),
        ("e", x+y, Integer(1)),
        ("pi", 1 + 2*(x+y), y-x),
        ("zeta(2)", 2*x**2 + 2*x*y + y**2, -2*x**2 + 2*x*y - y**2),
        ("zeta(3)", x**3 + 2*x**2*y + 2*x*y**2 + y**3, -x**3 + 2*x**2*y - 2*x*y**2 + y**3),
    ]
    
    for name, f, fbar in known:
        lin_ok = check_linear_condition(f, fbar)
        quad_ok = check_quadratic_condition(f, fbar)
        M_X, M_Y, a, b = build_cmf_matrices(f, fbar)
        cmf_ok = verify_conservation(M_X, M_Y)
        
        status = "✓" if (lin_ok and quad_ok and cmf_ok) else "✗"
        print(f"  {status} {name}: lin={lin_ok}, quad={quad_ok}, cmf={cmf_ok}")


if __name__ == "__main__":
    print("="*70)
    print("CMF Discovery via f,fbar Optimization")
    print("="*70)
    
    verify_known_cmfs()
    
    print("\n" + "="*70)
    print("Searching for CMFs (integer coefficients, coef_range=2)")
    print("="*70)
    
    for degree in [1, 2]:
        print(f"\nDegree {degree}:")
        solutions = find_all_cmfs(degree, method='integer', coef_range=2, 
                                  max_nonzero=4, verbose=True)
        
        # Filter to show only non-trivial (non-constant) solutions
        nontrivial = [s for s in solutions if s.f.has(x) or s.f.has(y)]
        print(f"\n  Found {len(nontrivial)} non-trivial CMF families")
