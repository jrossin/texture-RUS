from sympy import S, filldedent, Symbol, sympify, MatrixBase, Matrix, Expr, Eq, Add, Equality
from sympy.core.compatibility import is_sequence
from sympy.core.sympify import _sympify
from collections import defaultdict

class NonlinearError(Exception):
    pass

def linear_coeffs(eq, *syms, **_kw):
    """Return a list whose elements are the coefficients of the
    corresponding symbols in the sum of terms in  ``eq``.
    The additive constant is returned as the last element of the
    list.
    Raises
    ======
    NonlinearError
        The equation contains a nonlinear term
    Examples
    ========
    >>> linear_coeffs(3*x + 2*y - 1, x, y)
    [3, 2, -1]
    It is not necessary to expand the expression:
    >>> linear_coeffs(x + y*(z*(x*3 + 2) + 3), x)
    [3*y*z + 1, y*(2*z + 3)]
    But if there are nonlinear or cross terms -- even if they would
    cancel after simplification -- an error is raised so the situation
    does not pass silently past the caller's attention:
    >>> eq = 1/x*(x - 1) + 1/x
    >>> linear_coeffs(eq.expand(), x)
    [0, 1]
    >>> linear_coeffs(eq, x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term encountered: 1/x
    >>> linear_coeffs(x*(y + 1) - x*y, x, y)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term encountered: x*(y + 1)
    """
    d = defaultdict(list)
    eq = _sympify(eq)
    symset = set(syms)
    has = eq.free_symbols & symset
    if not has:
        return [S.Zero]*len(syms) + [eq]
    c, terms = eq.as_coeff_add(*has)
    d[0].extend(Add.make_args(c))
    for t in terms:
        m, f = t.as_coeff_mul(*has)
        if len(f) != 1:
            break
        f = f[0]
        if f in symset:
            d[f].append(m)
        elif f.is_Add:
            d1 = linear_coeffs(f, *has, **{'dict': True})
            d[0].append(m*d1.pop(0))
            for xf, vf in d1.items():
                d[xf].append(m*vf)
        else:
            break
    else:
        for k, v in d.items():
            d[k] = Add(*v)
        if not _kw:
            return [d.get(s, S.Zero) for s in syms] + [d[0]]
        return d  # default is still list but this won't matter
    raise NonlinearError('nonlinear term encountered: %s' % t)

def linear_eq_to_matrix(equations, *symbols):
    r"""
    Converts a given System of Equations into Matrix form.
    Here `equations` must be a linear system of equations in
    `symbols`. Element M[i, j] corresponds to the coefficient
    of the jth symbol in the ith equation.
    The Matrix form corresponds to the augmented matrix form.
    For example:
    .. math:: 4x + 2y + 3z  = 1
    .. math:: 3x +  y +  z  = -6
    .. math:: 2x + 4y + 9z  = 2
    This system would return `A` & `b` as given below:
    ::
         [ 4  2  3 ]          [ 1 ]
     A = [ 3  1  1 ]   b  =   [-6 ]
         [ 2  4  9 ]          [ 2 ]
    The only simplification performed is to convert
    `Eq(a, b) -> a - b`.
    Raises
    ======
    NonlinearError
        The equations contain a nonlinear term.
    ValueError
        The symbols are not given or are not unique.
    Examples
    ========
    >>> from sympy import linear_eq_to_matrix, symbols
    >>> c, x, y, z = symbols('c, x, y, z')
    The coefficients (numerical or symbolic) of the symbols will
    be returned as matrices:
    >>> eqns = [c*x + z - 1 - c, y + z, x - y]
    >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])
    >>> A
    Matrix([
    [c,  0, 1],
    [0,  1, 1],
    [1, -1, 0]])
    >>> b
    Matrix([
    [c + 1],
    [    0],
    [    0]])
    This routine does not simplify expressions and will raise an error
    if nonlinearity is encountered:
    >>> eqns = [
    ...     (x**2 - 3*x)/(x - 3) - 3,
    ...     y**2 - 3*y - y*(y - 4) + x - 4]
    >>> linear_eq_to_matrix(eqns, [x, y])
    Traceback (most recent call last):
    ...
    NonlinearError:
    The term (x**2 - 3*x)/(x - 3) is nonlinear in {x, y}
    Simplifying these equations will discard the removable singularity
    in the first, reveal the linear structure of the second:
    >>> [e.simplify() for e in eqns]
    [x - 3, x + y - 4]
    Any such simplification needed to eliminate nonlinear terms must
    be done before calling this routine.
    """
    if not symbols:
        raise ValueError(filldedent('''
            Symbols must be given, for which coefficients
            are to be found.
            '''))

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    for i in symbols:
        if not isinstance(i, Symbol):
            raise ValueError(filldedent('''
            Expecting a Symbol but got %s
            ''' % i))

    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    A, b = [], []
    for i, f in enumerate(equations):
        if isinstance(f, Equality):
            f = f.rewrite(Add, evaluate=False)
        coeff_list = linear_coeffs(f, *symbols)
        b.append(-coeff_list.pop())
        A.append(coeff_list)
    A, b = map(Matrix, (A, b))
    return A, b
