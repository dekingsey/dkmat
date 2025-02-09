import pytest
import numpy as np
from dkmat import *

def test_conversion_matrice_2x2():
    result = asmat(range(4))
    expected = np.array([[0, 1], [2, 3]])
    # Vérification que la matrice résultante est correcte
    assert np.array_equal(result, expected)

def test_conversion_matrice_2x3():
    result = asmat(range(6), 2)
    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.array_equal(result, expected)

def test_conversion_matrice_3x3():
    result = asmat(range(9))
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert np.array_equal(result, expected)

def test_exception_conversion_incorrecte():
    with pytest.raises(ValueError):
        asmat(range(7))

def test_conversion_matrice_1x1():
    result = asmat([42])
    expected = np.array([[42]])
    assert np.array_equal(result, expected)

def test_pprint(capsys):
    pprint(asmat(range(4))/11)
    captured = capsys.readouterr()
    expected_output = " 0   1/11 \n2/11 3/11 \n"
    assert captured.out == expected_output
