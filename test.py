import unittest
import logging
import numpy
import element

class TrainingTestCase(unittest.TestCase):

    def test_training(self):
        for training_func in [element.online_training, element.batch_training]:
            print("{}".format(training_func))
            for expected_outputs in self.BOOL_FUNCTIONS:
                input_count = len(expected_outputs[0][0])
                print("#len={}: {}".format(input_count, expected_outputs))
                for _ in range(1000): # generate some random elements
                    e = element.SchwellwertElement(input_count)
                    training_func(e, expected_outputs)
                    for test_input, expected_output in expected_outputs:
                        self.assertEqual(expected_output, e.ausgabe(test_input))


    BOOL_FUNCTIONS = [
        [
            # negation
            (numpy.array([0]), 1),
            (numpy.array([1]), 0)
        ],
        [   # OR 
            (numpy.array([0,0]), 0),
            (numpy.array([0,1]), 0),
            (numpy.array([1,0]), 0),
            (numpy.array([1,1]), 1)
        ],
        [   # AND
            (numpy.array([0,0]), 0),
            (numpy.array([0,1]), 1),
            (numpy.array([1,0]), 1),
            (numpy.array([1,1]), 1)
        ],
        [   # IMPLICATION
            (numpy.array([0,0]), 1),
            (numpy.array([0,1]), 1),
            (numpy.array([1,0]), 0),
            (numpy.array([1,1]), 1)
        ],
        [   # triple AND
            (numpy.array([0,0,0]), 0),
            (numpy.array([0,0,1]), 0),
            (numpy.array([0,1,0]), 0),
            (numpy.array([0,1,1]), 0),
            (numpy.array([1,0,0]), 0),
            (numpy.array([1,0,1]), 0),
            (numpy.array([1,1,0]), 0),
            (numpy.array([1,1,1]), 1)
        ],
    ]
