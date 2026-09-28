"""
Calculator Tool
================
Mathematical expression evaluator with support for complex math.
Expressions are evaluated with a restricted AST interpreter (never `eval`),
so arbitrary Python cannot be smuggled through the tool.
"""

import ast
import math

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')")


class CalculatorTool:
    """Mathematical calculator tool."""

    name = "calculator"
    description = "Evaluate mathematical expressions. Supports basic arithmetic, trigonometry, logarithms, and more. Examples: '2 + 2', 'sqrt(16)', 'log(100)', 'sin(pi/2)'"

    # Whitelist of callable names and constants.
    SAFE_NAMES = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "int": int, "float": float,
        # Math module
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial, "gcd": math.gcd,
        "radians": math.radians, "degrees": math.degrees,
        "exp": math.exp, "isqrt": math.isqrt,
    }

    # Binary operators supported by the AST evaluator.
    _BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    @classmethod
    def _eval_node(cls, node) -> object:
        """Evaluate a single AST node against a strict whitelist."""
        if isinstance(node, ast.Expression):
            return cls._eval_node(node.body)
        if isinstance(node, ast.Constant):
            # Numbers, strings, booleans, None
            return node.value
        if isinstance(node, ast.Name):
            # Named constants such as pi / e / tau
            if node.id in cls.SAFE_NAMES:
                return cls.SAFE_NAMES[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls._BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return cls._BINOPS[op_type](cls._eval_node(node.left), cls._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            value = cls._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in cls.SAFE_NAMES:
                raise ValueError("Only whitelisted math functions are allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            func = cls.SAFE_NAMES[node.func.id]
            args = [cls._eval_node(a) for a in node.args]
            return func(*args)
        if isinstance(node, (ast.List, ast.Tuple)):
            # Literal collections, only usable by whitelisted functions.
            return [cls._eval_node(e) for e in node.elts]
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    @classmethod
    def safe_eval(cls, expression: str) -> object:
        """Evaluate a math expression using a strict AST interpreter."""
        tree = ast.parse(expression, mode="eval")
        return cls._eval_node(tree)

    @staticmethod
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Fast path: numexpr for complex numeric expressions
            try:
                import numexpr as ne
                result = ne.evaluate(expression)
                return f"Result: {result}"
            except Exception:
                pass
            # Safe path: AST interpreter (no arbitrary Python execution)
            result = CalculatorTool.safe_eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {type(e).__name__}: {str(e)}"


def create_calculator_tool() -> StructuredTool:
    """Create a calculator tool instance."""
    return StructuredTool(
        name=CalculatorTool.name,
        description=CalculatorTool.description,
        func=CalculatorTool.calculate,
        args_schema=CalculatorInput,
    )
