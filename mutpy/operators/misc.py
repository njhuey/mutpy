import ast

from mutpy import utils
from mutpy.operators.arithmetic import AbstractArithmeticOperatorReplacement
from mutpy.operators.base import MutationOperator, MutationResign, copy_node


class PandasMutator(MutationOperator):
    @copy_node
    def mutate_Call_argument(self, node):
        # Look for keyword arguments with the target name
        for keyword in node.keywords:
            if keyword.arg == "inplace":
                # Replace the value with the new value
                v = not keyword.value
                keyword.value = ast.Constant(value=v)
                return node  # Return the modified node
        # No matching argument found, skip mutation
        raise MutationResign()

    @copy_node
    def mutate_Call_reset_index(self, node):
        # Look for keyword arguments with the target name
        if isinstance(node.func, ast.Attribute) and node.func.attr == "reset_index":
            return node.func.value
        raise MutationResign()

    @copy_node
    def mutate_Return_reset_index(self, node):
        if isinstance(node.value, ast.Call):
            if node.value.func.attr != "reset_index":
                sum = ast.Call(func=node.value.func, args=[], keywords=[])
                reset_index = ast.Attribute(
                    value=sum, attr="reset_index", ctx=ast.Load()
                )
                return ast.Return(
                    value=ast.Call(func=reset_index, args=[], keywords=[])
                )
        raise MutationResign()

    @copy_node
    def mutate_Call_keep_dims(self, node):
        # Look for keyword arguments with the target name
        for keyword in node.keywords:
            if keyword.arg == "keepdims":
                # Replace the value with the new value
                v = not keyword.value.value
                keyword.value = ast.Constant(value=v)
                return node  # Return the modified node
        # No matching argument found, skip mutation
        raise MutationResign()

    @copy_node
    def mutate_Call_argument_axis(self, node):
        # Look for keyword arguments with the target name
        for keyword in node.keywords:
            if keyword.arg == "axis":
                # Replace the value with the new value
                v = keyword.value.value ^ 1
                keyword.value = ast.Constant(value=v)
                return node  # Return the modified node
        # No matching argument found, skip mutation
        raise MutationResign()
    
    @classmethod
    def name(cls):
        return "PDM"

class TypeChanger(MutationOperator):
    # 1280
    @copy_node
    def mutate_Call_float64attr(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "float64":
            node.func.attr = "float32"
            return node
        raise MutationResign()

    # 403, 405, 1934
    @copy_node
    def mutate_Assign_float(self, node):
        if isinstance(node.value, ast.IfExp):
            if isinstance(node.value.orelse, ast.Attribute):
                if node.value.orelse.attr == "float64":
                    node.value.orelse.attr = "float32"
                    return node
                elif node.value.orelse.attr == "float32":
                    node.value.orelse.attr = "float64"
                    return node
        raise MutationResign()

    # 400, 1053
    @copy_node
    def mutate_Call_float64arg(self, node):
        if len(node.args) > 0:
            if (
                isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "float64"
            ):
                node.args[0].attr = "float32"
                return node
        raise MutationResign()

    # 1248
    @copy_node
    def mutate_If_float64(self, node):
        if len(node.body) > 0:
            if isinstance(node.body[0], ast.Assign):
                if isinstance(node.body[0].value, ast.IfExp):
                    if isinstance(node.body[0].value.body, ast.Attribute):
                        if node.body[0].value.body.attr == "float64":
                            node.body[0].value.body.attr = "float32"
                            return node
        raise MutationResign()

    # 403, 405, 1934
    @copy_node
    def mutate_Assign_complex64(self, node):
        if isinstance(node.value, ast.IfExp):
            if isinstance(node.value.body, ast.Attribute):
                if node.value.body.attr == "complex64":
                    node.value.body.attr = "complex128"
                    return node
                elif node.value.body.attr == "complex128":
                    node.value.body.attr = "complex64"
                    return node
        raise MutationResign()

    # 1248
    @copy_node
    def mutate_If_complex128(self, node):
        if len(node.body) > 0:
            if isinstance(node.body[0], ast.Assign):
                if isinstance(node.body[0].value, ast.IfExp):
                    if isinstance(node.body[0].value.orelse, ast.Attribute):
                        if node.body[0].value.orelse.attr == "complex128":
                            node.body[0].value.orelse.attr = "complex64"
                            return node
        raise MutationResign()

    # 1280
    @copy_node
    def mutate_Call_complex128attr(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "complex128":
            node.func.attr = "complex64"
            return node
        raise MutationResign()

    @classmethod
    def name(cls):
        return "TCH"


class NumpyMutator(MutationOperator):
    @copy_node
    def mutate_Call_any2all(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "any":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="all", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_all2any(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "all":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="any", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_zeros2ones(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "zeros":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="ones", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_ones2zeros(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ones":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="zeros", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_average2mean(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "average":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="mean", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_zeros2zeros_like(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "zeros":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="zeros_like", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_zeros_like2zeros(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "zeros_like":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="zeros", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_ones2ones_like(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ones":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="ones_like", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @copy_node
    def mutate_Call_ones_like2ones(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ones_like":
            name = node.func.value
            all_node = ast.Attribute(value=name, attr="ones", ctx=ast.Load())
            node.func = all_node
            return node
        raise MutationResign()

    @classmethod
    def name(cls):
        return "NPM"


class AssignmentOperatorReplacement(AbstractArithmeticOperatorReplacement):
    def should_mutate(self, node):
        return isinstance(node.parent, ast.AugAssign)

    @classmethod
    def name(cls):
        return "ASR"


class BreakContinueReplacement(MutationOperator):
    def mutate_Break(self, node):
        return ast.Continue()

    def mutate_Continue(self, node):
        return ast.Break()


class ConstantReplacement(MutationOperator):
    FIRST_CONST_STRING = "mutpy"
    SECOND_CONST_STRING = "python"

    def mutate_Constant(self, node):
        if isinstance(node.value, (int, float)):  # Handle numbers
            return ast.Constant(value=node.value + 1)

        elif isinstance(node.value, str):  # Handle strings
            if utils.is_docstring(node):
                raise MutationResign()

            if node.value != self.FIRST_CONST_STRING:
                return ast.Constant(value=self.FIRST_CONST_STRING)
            else:
                return ast.Constant(value=self.SECOND_CONST_STRING)

        elif node.value is not None:  # Handle NameConstant (True, False, None)
            return ast.Constant(value=None)

        raise MutationResign()  # Skip mutation if the constant isn't relevant

    def mutate_Constant_empty(self, node):
        if isinstance(node.value, str) and node.value and not utils.is_docstring(node):
            return ast.Constant(
                value=""
            )  # Replace non-empty strings with empty strings

        raise MutationResign()  # Skip for non-string nodes or docstrings
        return ast.Str(s="")

    @classmethod
    def name(cls):
        return "CRP"


class DefaultParameterMutation(MutationOperator):
    """Mutate default parameter values based on parameter type."""

    def mutate_arguments_defaults(self, node):
        """Remove the default arguments from a function."""
        if isinstance(node, ast.arguments) and len(node.defaults) != 0:
            node.defaults = []
            return node
        raise MutationResign()

    def mutate_arguments_default_to_and_from_None(self, node):
        """
        Mutate default arguments to and from `None`.

        For all default arguments that are `None`, mutate default argument value to a
        string with a low collision chance. If a default argument is not `None`, change
        it to `None`.
        """
        if isinstance(node, ast.arguments) and len(node.defaults) != 0:
            for i, default_value in enumerate(node.defaults):
                if default_value is None:
                    # Randomly generated 16 character string to avoid collisions.
                    node.defaults[i].value = "GxMEuCGdW4Lm75gr"
                else:
                    node.defaults[i].value = None
            return node
        raise MutationResign()

    def mutate_arguments_default_from_None(self, node):
        """Mutate a `None` default argument to not `None`."""
        if (
            isinstance(node.parent, ast.arguments)
            and isinstance(node, ast.Constant)
            and node.value is None
        ):
            # Randomly generated 16 character string to avoid collisions.
            node.value = "GxMEuCGdW4Lm75gr"
            return node
        raise MutationResign()

    @classmethod
    def name(cls):
        return "DPM"


class SliceIndexRemove(MutationOperator):
    def mutate_Slice_remove_lower(self, node):
        if not node.lower:
            raise MutationResign()

        return ast.Slice(lower=None, upper=node.upper, step=node.step)

    def mutate_Slice_remove_upper(self, node):
        if not node.upper:
            raise MutationResign()

        return ast.Slice(lower=node.lower, upper=None, step=node.step)

    def mutate_Slice_remove_step(self, node):
        if not node.step:
            raise MutationResign()

        return ast.Slice(lower=node.lower, upper=node.upper, step=None)


class SelfVariableDeletion(MutationOperator):
    def mutate_Attribute(self, node):
        try:
            if node.value.id == "self":
                return ast.Name(id=node.attr, ctx=ast.Load())
            else:
                raise MutationResign()
        except AttributeError:
            raise MutationResign()


class StatementDeletion(MutationOperator):
    def mutate_Assign(self, node):
        return ast.Pass()

    def mutate_Return(self, node):
        return ast.Pass()

    def mutate_Expr(self, node):
        if utils.is_docstring(node.value):
            raise MutationResign()
        return ast.Pass()

    @classmethod
    def name(cls):
        return "SDL"
