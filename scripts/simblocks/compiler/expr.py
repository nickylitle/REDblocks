"""Safe expressions for system files.

'= sqrt(2 * 9.81 * $h)'           numeric expression over params ($name) and math functions (SI values)
'hub + $R_tip * rotor - target'    linear measure over BODY coordinates (used by elements)

Only arithmetic, numbers, $params, body names and a whitelist of math functions are allowed. Anything else
(attribute access, calls to other names, imports) is rejected by an AST check before evaluation."""
import ast
import math
import re

FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log",
                                       "log10", "radians", "degrees", "hypot")}
FUNCS.update(abs=abs, min=min, max=max, pi=math.pi, e=math.e)
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
            ast.BoolOp, ast.And, ast.Or, ast.IfExp)
_PARAM = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def _mangle(name):
    return "__p_" + name.replace(".", "__")


class ExprError(ValueError):
    pass


class ExprZeroDivision(ExprError, ZeroDivisionError):
    """A division by zero in an expression: an ExprError with its place, and still a ZeroDivisionError for the
    callers that turn it into infinity (a safety factor at zero demand)."""


def _check(tree, allowed_names):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ExprError(f"'{type(node).__name__}' is not allowed in expressions")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise ExprError("only math functions (sqrt, sin, cos, ...) may be called")
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in FUNCS:
            raise ExprError(f"unknown name '{node.id}'")


def params_in(text):
    return set(_PARAM.findall(text))


def eval_numeric(text, params):
    """Evaluate '= ...' (leading '=' optional) with $params substituted."""
    src = text.strip().lstrip("=").strip()
    missing = [p for p in params_in(src) if p not in params]
    if missing:
        raise ExprError(f"undefined parameter(s): {', '.join('$' + m for m in missing)}")
    src = _PARAM.sub(lambda m: _mangle(m.group(1)), src)
    names = {_mangle(k): v for k, v in params.items()}
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot read the expression '{text.strip()}': {e.msg}" +
                        (f" near column {e.offset}" if e.offset else "")) from None
    _check(tree, set(names))
    try:
        r = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {**FUNCS, **names})
        if isinstance(r, int) and not isinstance(r, bool):
            r = float(r)                       # 2 ** 10000 overflows here, inside the try
        return r
    except ZeroDivisionError:
        raise ExprZeroDivision(f"'{text.strip()}' divides by zero") from None
    except (ArithmeticError, ValueError, TypeError) as e:
        raise ExprError(f"cannot evaluate '{text.strip()}': {e}") from None


class Lin:
    """Linear combination of body coordinates plus a constant."""
    def __init__(self, coef=None, const=0.0):
        self.coef = dict(coef or {}); self.const = float(const)

    def _lift(self, o):
        return o if isinstance(o, Lin) else Lin(const=float(o))

    def __add__(self, o):
        o = self._lift(o); c = dict(self.coef)
        for k, v in o.coef.items():
            c[k] = c.get(k, 0.0) + v
        return Lin(c, self.const + o.const)
    __radd__ = __add__

    def __neg__(self):
        return Lin({k: -v for k, v in self.coef.items()}, -self.const)

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        if isinstance(o, Lin):
            if o.coef and self.coef:
                raise ExprError("a measure must be LINEAR in body coordinates (body * body is not allowed)")
            return self.__mul__(o.const) if self.coef else o.__mul__(self.const)
        return Lin({k: v * o for k, v in self.coef.items()}, self.const * o)
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Lin):
            if o.coef:
                raise ExprError("cannot divide by a body coordinate")
            o = o.const
        return self * (1.0 / o)


def eval_measure(text, bodies, params):
    """Linear measure over body names, e.g. 'hub + $R_tip*rotor - target' -> ({hub:1, rotor:R, target:-1}, const)."""
    src = str(text).strip()
    missing = [p for p in params_in(src) if p not in params]
    if missing:
        raise ExprError(f"undefined parameter(s): {', '.join('$' + m for m in missing)}")
    src = _PARAM.sub(lambda m: _mangle(m.group(1)), src)
    names = {_mangle(k): v for k, v in params.items()}
    names.update({b: Lin({b: 1.0}) for b in bodies})
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot read the measure '{str(text).strip()}': {e.msg}") from None
    _check(tree, set(names))
    try:
        out = eval(compile(tree, "<measure>", "eval"), {"__builtins__": {}}, {**FUNCS, **names})
    except ExprError:
        raise
    except ZeroDivisionError:
        raise ExprError(f"the measure '{str(text).strip()}' divides by zero") from None
    except (ArithmeticError, ValueError, TypeError):
        raise ExprError("a measure must be LINEAR in the body coordinates (sums of number x body, plus a constant): "
                        f"'{str(text).strip()}' is not") from None
    if not isinstance(out, Lin):
        raise ExprError("a measure must contain at least one body name")
    return {k: v for k, v in out.coef.items() if v != 0.0}, out.const
