#
# Lispy: A lisp interpreter in Python
#

import sys
import re
import StringIO

LEFT_PAREN = '('
RIGHT_PAREN = ')'
TOKENIZER = r'''\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*)(.*)'''
TRUE = "#t"
FALSE = "#f"


class Symbol(str):
    pass


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
        if isinstance(params, Symbol):
            self.update({params: list(args)})
            print {params: list(args)}
        else:
            self.update(zip(params, args))
        self.parent = parent

    def find(self, var):
        if var in self:
            return self
        else:
            return self.parent.find(var)


def Sym(s, symbol_table={}):
    """
    Find or create unique symbol entry for 's' in
    a symbol table
    """
    if s not in symbol_table:
        symbol_table[s] = Symbol(s)

    return symbol_table[s]


EOF = Symbol('#<eof-object>')


class InPort(object):
    """Input port.

    Used to retain a line of characters.
    """
    def __init__(self, file):
        self.file = file
        self.line = ''

    def next_token(self):
        """Return next token

        As a consequence, new chars will be stored into
        the line buffer(if needed).
        """
        if self.line == '':
            self.line = self.file.readline()

        if self.line == '':
            return EOF

        token, self.line = re.match(TOKENIZER, self.line).groups()

        if token and not token.startswith(';'):
            return token


QUOTE, IF, SET, DEFINE, LAMBDA, BEGIN, DEFINEMACRO, = map(
    Sym, "quote   if   set!  define   lambda   begin   define-macro".split()
)

QUASIQUOTE, UNQUOTE, UNQUOTESPLICING = map(
    Sym, "quasiquote   unquote   unquote-splicing".split()
)

APPEND, CONS = map(Sym, "append cons".split())

QUOTES = {
    "'": QUOTE,
    "`": QUASIQUOTE,
    ",": UNQUOTE,
    ",@": UNQUOTESPLICING
}


def callcc(proc):
    """Call proc with continuation

    Note: This works for simple escape only.

    For more info on call with continuations, refer
    http://community.schemewiki.org/?call-with-current-continuation
    """
    warning = RuntimeWarning("Continuation called")

    def throw(val):
        warning.retval = val
        raise warning

    try:
        return proc(throw)
    except RuntimeWarning as w:
        if w is warning:
            return warning.retval
        else:
            raise w


def arithmetic(op):
    "Perform arithmetic operation on multiple arguments"
    def wrapper(*args):
        if op == '+':
            return sum(args)
        elif op == '-':
            result = args[0]
            for arg in args[1:]:
                result = result - arg
            return result
        elif op == '*':
            result = 1
            for arg in args:
                result = arg * result
            return result
        elif op == '/':
            result = args[0]
            for arg in args[1:]:
                result = result / float(arg)
            return result
    return wrapper


def standard_env():
    "An environment with some Scheme standard procedures."
    import math
    import operator as op
    env = Env()
    # Add math functions to env
    env.update(vars(math))
    env.update({
        '+': arithmetic('+'),
        '-': arithmetic('-'),
        '*': arithmetic('*'),
        '/': arithmetic('/'),
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
        'symbol?': lambda x: isinstance(x, Symbol),
        'call/cc': callcc
    })
    return env


global_env = standard_env()


def tokenize(string):
    "Returns a list of all tokens in given string"
    return string.replace('(', '( ').replace(')', ' )').split()


def readchar(inport):
    "Read next character from inport"
    if inport.line:
        ch, inport.line = inport.line[0], inport.line[1:]
        return ch
    else:
        return inport.file.read(1) or EOF


def read(inport):
    "Read a Scheme expression from input port"
    def read_ahead(token):
        if token == LEFT_PAREN:
            sub_exp = []

            while True:
                token = inport.next_token()
                if token == RIGHT_PAREN:
                    return sub_exp
                else:
                    sub_exp.append(read_ahead(token))
        elif token == RIGHT_PAREN:
            raise SyntaxError("unexpected ')'")
        elif token in QUOTES:
            return [QUOTES[token], read(inport)]
        elif token == EOF:
            raise SyntaxError("unexpected eof in list")
        else:
            return atom(token)

    token_next = inport.next_token()
    return EOF if token_next is EOF else read_ahead(token_next)


def NUMBER(token):
    "Parse token as a number"
    try:
        return int(token)
    except ValueError:
        pass

    try:
        return float(token)
    except ValueError:
        pass

    return complex(token.replace('i', 'j', 1))


def atom(token):
    "Parse token as a boolean/string/number or a symbol"
    if token == TRUE:
        return True
    elif token == FALSE:
        return False
    elif token == '"':
        return token[1:-1].decode('string_escape')

    try:
        return NUMBER(token)
    except ValueError:
        return Sym(token)


def parse(inport):
    "Parse a scheme expression from input port"
    if isinstance(inport, str):
        # if input is simply a string, convert it
        # to an inport compatible form
        inport = InPort(StringIO.StringIO(inport))
    return expand(read(inport), toplevel=True)


def eval(exp, env=global_env):
    "Evaluates a expression in context of \
    an environment"
    while True:
        if isinstance(exp, Symbol):
            return env.find(exp)[exp]
        elif not isinstance(exp, list):
            # Return as is, if not an expression
            return exp
        elif exp[0] == QUOTE:
            (_, _exp) = exp
            return _exp
        elif exp[0] == DEFINE:
            (_, variable, value) = exp
            env[variable] = eval(value, env)
            return None
        elif exp[0] == IF:
            (_, test, conseq, alt) = exp
            exp = conseq if eval(test, env) else alt
        elif exp[0] == SET:
            (_, var, val) = exp
            env.find(var)[var] = eval(val, env)
            return None
        elif exp[0] == LAMBDA:
            (_, params, body) = exp
            return Procedure(params, body, env)
        elif exp[0] == BEGIN:
            for _exp in exp[1:-1]:
                eval(_exp, env)
                exp = exp[-1]
        else:
            # Parse a procedure call
            args = [eval(arg, env) for arg in exp]
            proc = args.pop(0)
            if isinstance(proc, Procedure):
                # Using exp and env together as accumulators
                # to remove recursion limit
                exp = proc.body
                env = Env(proc.params, args, proc.env)
            else:
                return proc(*args)


def require(exp, predicate, message=None):
    "Raise a syntax error if predicate is false"
    if not predicate:
        raise SyntaxError(schemeify(exp) + " => " + message)


def is_pair(x):
    return x != [] and isinstance(x, list)


def expand_quasiquote(exp):
    """Expand `exp => 'exp; `,exp => exp; `(,@exp y) => (append exp y) """
    if not is_pair(exp):
        return [QUOTE, exp]
    require(exp, exp[0] is not UNQUOTESPLICING, "can't splice here")
    if exp[0] is UNQUOTE:
        require(exp, len(exp) == 2)
        return exp[1]
    elif is_pair(exp[0]) and exp[0][0] is UNQUOTESPLICING:
        require(exp[0], len(exp[0]) == 2)
        return [APPEND, exp[0][1], expand_quasiquote(exp[1:])]
    else:
        return [CONS,
                expand_quasiquote(exp[0]),
                expand_quasiquote(exp[1:])]


def expand(exp, toplevel=False):
    "Walk tree of exp, making optimizations/fixes, and signaling SyntaxError."
    require(exp, exp != [])                    # () => Error

    if not isinstance(exp, list):                 # constant => unchanged
        return exp

    elif exp[0] is QUOTE:                 # (quote exp)
        require(exp, len(exp) == 2)
        return exp

    elif exp[0] is IF:
        if len(exp) == 3:
            exp = exp + [None]     # (if t c) => (if t c None)

        require(exp, len(exp) == 4)
        return map(expand, exp)

    elif exp[0] is SET:
        require(exp, len(exp) == 3)
        var = exp[1]                       # (set! non-var exp) => Error
        require(exp, isinstance(var, Symbol), "can set! only a symbol")
        return [SET, var, expand(exp[2])]

    elif exp[0] is DEFINE:
        require(exp, len(exp) >= 3)
        _def, v, body = exp[0], exp[1], exp[2:]

        if isinstance(v, list) and v:           # (define (f args) body)
            f, args = v[0], v[1:]        # => (define f (lambda (args) body))
            return expand([_def, f, [LAMBDA, args]+body])
        else:
            # (define non-var/list eexpp) => Error
            require(exp, len(exp) == 3, "3 args expected")
            require(exp, isinstance(v, Symbol), "can define only a symbol")
            _exp = expand(exp[2])
            return [DEFINE, v, _exp]

    elif exp[0] is BEGIN:
        if len(exp) == 1:
            return None        # (begin) => None
        else:
            return [expand(expi, toplevel) for expi in exp]

    elif exp[0] is LAMBDA:                # (lambda (exp) e1 e2)
        # => (lambda (exp) (begin e1 e2))
        require(exp, len(exp) >= 3)
        vars, body = exp[1], exp[2:]

        check1 = (isinstance(vars, list) and
                  all(isinstance(v, Symbol) for v in vars))
        check2 = isinstance(vars, Symbol)
        pred = check1 or check2
        require(exp, pred, "illegal lambda argument list")

        exp = body[0] if len(body) == 1 else [BEGIN] + body
        return [LAMBDA, vars, expand(exp)]

    elif exp[0] is QUASIQUOTE:            # `exp => expand_quasiquote(exp)
        require(exp, len(exp) == 2)
        return expand_quasiquote(exp[1])

    else:
        return map(expand, exp)            # (f arg...) => expand each


def schemeify(val):
    "Convert value to a valid scheme string"
    if val is True:
        return '#t'
    elif val is False:
        return '#f'
    elif isinstance(val, Symbol):
        return val
    elif isinstance(val, str):
        val = val.encode('string_escape').replace('"', r'\"')
        return '"{}"'.format(val)
    elif isinstance(val, complex):
        val = str(val).replace('j', 'i')
        return '"{}"'.format(val)
    elif isinstance(val, list):
        elems = ' '.join([str(x) for x in val])
        return ''.join(['(', elems, ')'])
    else:
        return str(val)


def load(filename):
    "Eval scheme code from a file"
    repl(None, InPort(open(filename)), None)


def repl(prompt="lispy> ", inport=InPort(sys.stdin), out=sys.stdout):
    "REPL for lispy"
    sys.stderr.write("Lispy v1.0\n")
    while True:
        try:
            if prompt:
                sys.stderr.write(prompt)

            parsed_val = parse(inport)
            if parsed_val is EOF:
                return

            result = eval(parsed_val)
            if result and out:
                print >> out, schemeify(result)
        except Exception as e:
            import traceback
            print traceback.format_exc()
            # print "{}: {}".format(type(e).__name__, e)


if __name__ == '__main__':
    repl()
