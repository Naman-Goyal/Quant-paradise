from os.path import dirname as dir
from sys import path
from sys import path
import os
import sys
from Mod6 import*
path.append(dir(path[0])[0:dir(path[0]).rfind('\\')] + r'\Vaeick_CIR')
path.append(
    '/home/travis/build/Naman-Goyal/Quant-paradise/PricingFIP/' + 
    'Vaeick_CIR')


def test_first():

    assert zero_coupon(1, 2, 3, 4, 5, model="abcd") == -1


def test_second():

    assert swapRates([1], 2, 3) == -1


def test_third():

    assert liborRates([1], 2, 3) == -1


def test_fourth():

    assert objFunc1([1, 2, 3, -8], 5, 6, 2, model="abcd") == -2


def test_fifth():

    assert objFunc1([-1, 2, 3, 4], 2, 3, 4, model="ac") == -1
