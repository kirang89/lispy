#
# Lispy: A lisp interpreter in Python
#

LEFT_PAREN = '('
RIGHT_PAREN = ')'
QUOTE = 'quote'
DEFINITION = 'define'
CONDITIONAL = 'if'
SYMBOL = str
SET = 'set!'
LAMBDA = 'lambda'


class Procedure(object):
    "Data structure to represent a procedure in Scheme"
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env

    def __call__(self, *args):
        env = Env(self.params, args, self.env)
        return eval(self.body, env)


class Env(dict):
    """
    An environment to provide context for various
    operations by the interpreter
    """
    def __init__(self, params=(), args=(), parent=None):
        self.update(zip(params, args))
        self.parent = parent

    def find(self, var):
        if var in self:
            return self
        else:
            return self.parent.find(var)


def standard_env():
    "An environment with some Scheme standard procedures."
    import math
    import operator as op
    env = Env()
    # Add math functions to env
    env.update(vars(math))
    env.update({
        '+': op.add,
        '-': op.sub,
        '*': op.mul,
        '/': op.div,
        '>': op.gt,
        '<': op.lt,
        '>=': op.ge,
        '<=': op.le,
        '=': op.eq,
        'abs': abs,
        'append': op.add,
        'apply': apply,
        'begin': lambda *x: x[-1],
        'car': lambda x: x[0],
        'cdr': lambda x: x[1:],
        'cons': lambda x, y: [x] + y,
        'eq?': op.is_,
        'equal?': op.eq,
        'length': len,
        'list': lambda *x: list(x),
        'list?': lambda x: isinstance(x, list),
        'map': map,
        'max': max,
        'min': min,
        'not': op.not_,
        'null?': lambda x: x == [],
        'number?': lambda x: isinstance(x, (int, float)),
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, SYMBOL),
    })
    return env


global_env = standard_env()


def tokenize(string):
    "Returns a list of all tokens in given string"
    return string.replace('(', '( ').replace(')', ' )').split()


def read_from_tokens(tokens):
    "Read an expression from a sequence of tokens"
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF while reading")

    token = tokens.pop(0)

    if token == LEFT_PAREN:
        sub_exp = []

        while tokens[0] != RIGHT_PAREN:
            sub_exp.append(read_from_tokens(tokens))

        tokens.pop(0)
        return sub_exp

    elif token == RIGHT_PAREN:
        raise SyntaxError("unexpected ')'")

    else:
        return atom(token)


def NUMBER(token):
    "Parse token as a number"
    try:
        return int(token)
    except ValueError:
        return float(token)


def atom(token):
    "Parse token as a number or a symbol"
    try:
        return NUMBER(token)
    except ValueError:
        return SYMBOL(token)


def parse(program):
    "Parse a scheme expression from a string"
    return read_from_tokens(tokenize(program))


def eval(exp, env=global_env):
    "Evaluates a expression in context of \
    an environment"
    if isinstance(exp, SYMBOL):
        return env.find(exp)[exp]
    elif not isinstance(exp, list):
        # Return as is, if not an expression
        return exp
    elif exp[0] == QUOTE:
        (_, _exp) = exp
        return _exp
    elif exp[0] == DEFINITION:
        (_, variable, value) = exp
        env[variable] = eval(value, env)
    elif exp[0] == CONDITIONAL:
        (_, test, conseq, alt) = exp
        _exp = conseq if eval(test, env) else alt
        return eval(_exp, env)
    elif exp[0] == SET:
        (_, var, val) = exp
        env.find(var)[var] = val
    elif exp[0] == LAMBDA:
        (_, params, body) = exp
        return Procedure(params, body, env)
    else:
        # Parse a procedure call
        procedure = eval(exp[0], env)
        args = [eval(arg, env) for arg in exp[1:]]
        return procedure(*args)


def schemeify(val):
    "Convert value to a valid scheme string"
    if isinstance(val, list):
        elems = ' '.join([str(x) for x in val])
        return ''.join(['(', elems, ')'])
    else:
        return str(val)


def repl(prompt="lispy> "):
    "REPL for lispy"
    while True:
        input = raw_input(prompt)
        if input == 'exit':
            return

        result = eval(parse(input))
        if result:
            print schemeify(result)
