# This code was based on https://github.com/Octavian-ai/english2cypher
# Octavian.ai developed.

import unittest
import tensorflow as tf

from .util import *

class TestUtil(unittest.TestCase):

    def test_detokenize_specials(self):
        """Test the detokenize_specials function in util.py."""
        s = "Hello<eos><eos>dsfsdfjdlsk<eos>sdfsfsfs"
        self.assertEqual(detokenize_specials(s), "Hello")

    def test_pretokenize_general(self):
        """Test the pretokenize_general function in util.py."""
        text = "How are you?"
        answer = "How <space> are <space> you?"
        self.assertEqual(pretokenize_general(text), answer)

    def test_multsub(self):
        """Thest the multsub function in util.py."""
        subs = "Wa Ha Ha"
        #new_list = [re.sub("[:\-() ]","_",x) for x in orig_list]

    def test_pretokenize_english(self):
        """Thest the test_pretokenize_english function in util.py."""
        text = "How are you?"
        answer = "How <space> are <space> you ? "
        self.assertEqual(pretokenize_english(text), answer)

    def test_pretokenize_cypher(self):
        """Thest the test_pretokenize_cypher function in util.py."""
        text = "MATCH (var1) = shortestPath((var1)-[*]-(var2))"
        answer = "MATCH <space>  ( var1 )  <space>  =  <space> shortestPath (  ( var1 )  -  [ * ]  -  ( var2 )  ) "
        self.assertEqual(pretokenize_cypher(text), answer)

    def test_expand_unknown_vocab(self):
        """Thest the expand_unknown_vocab function in util.py."""
        line = "How <space> is hjiof ? "
        vocab = ["How", "is", "<space>", "?"]
        answer = "How <space> is <h> <j> <i> <o> <f>  ? "
        self.assertEqual(expand_unknown_vocab(line, vocab), answer)


if __name__ == '__main__':
    unittest.main()
