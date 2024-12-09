"""
Calculator service for evaluating mathematical expressions.
"""

import ast
import operator
from typing import Union
from ..exceptions import CalculatorError


class Calculator:
    """A safe calculator that evaluates basic mathematical expressions."""

    # Supported operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,  # Unary minus
    }

    @staticmethod
    def evaluate(expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: A string containing a mathematical expression (e.g., "2 + 3 * 4")

        Returns:
            The result of evaluating the expression

        Raises:
            CalculatorError: If the expression is invalid or contains unsupported operations
        """
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode="eval")

            # Verify it's a simple mathematical expression
            if not isinstance(tree.body, (ast.BinOp, ast.UnaryOp, ast.Num)):
                raise CalculatorError(
                    "Expression too complex or contains unsupported operations"
                )

            return Calculator._eval_node(tree.body)

        except (SyntaxError, TypeError, ZeroDivisionError) as e:
            raise CalculatorError(f"Invalid expression: {str(e)}")
        except Exception as e:
            raise CalculatorError(f"Error evaluating expression: {str(e)}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate an AST node.

        Args:
            node: An AST node representing part of the expression

        Returns:
            The result of evaluating the node

        Raises:
            CalculatorError: If the node type is not supported
        """
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in cls.operators:
                raise CalculatorError("Unsupported operator")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -cls._eval_node(node.operand)
        else:
            raise CalculatorError("Unsupported operation in expression")
