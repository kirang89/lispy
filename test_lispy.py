#
# Test lispy implementation
#

import unittest
from lispy import parse, eval, schemeify


class LispyTest(unittest.TestCase):

    def setUp(self):
        "Setup code here"
        pass

    def tearDown(self):
        "Teardown code here"
        pass

    def test_define_exp_parse(self):
        exp = "(define a 10)"
        result = ['define', 'a', 10]
        self.assertEqual(parse(exp), result)

    def test_bad_exp_parse(self):
        exp = ")"
        with self.assertRaises(SyntaxError):
            parse(exp)

    def test_define_subexp_parse(self):
        exp = "(define a (+ 2 4))"
        result = ['define', 'a', ['+', 2, 4]]
        self.assertEqual(parse(exp), result)

    def test_eval_definition(self):
        exp = "(define a 10)"
        env = {}
        result = eval(parse(exp), env)
        self.assertIsNone(result)
        self.assertIn('a', env.keys())
        self.assertEqual(env['a'], 10)

    def test_eval_definition_and_expression(self):
        exp1 = "(define a 10)"
        exp2 = "(* a (+ 2 (- 4 2)))"
        result1 = eval(parse(exp1))
        self.assertIsNone(result1)
        result2 = eval(parse(exp2))
        self.assertEqual(result2, 40)

    def test_schemeify_list(self):
        input = [1, 2, 3]
        result = schemeify(input)
        self.assertEqual(result, '(1 2 3)')

    def test_schemeify_number(self):
        input = 123
        result = schemeify(input)
        self.assertEqual(result, '123')

    def test_lambda(self):
        exp1 = "(define square (lambda x (* x x)))"
        eval(parse(exp1))
        exp2 = "(square 10)"
        result = eval(parse(exp2))
        self.assertEqual(result, 100)

    def test_recursive_function(self):
        exp1 = "(define repeat (lambda f (lambda x (f (f x)))))"
        exp2 = "(define square (lambda x (* x x)))"
        exp3 = "((repeat square) 2)"
        eval(parse(exp1))
        eval(parse(exp2))
        result = eval(parse(exp3))
        self.assertEqual(result, 16)
