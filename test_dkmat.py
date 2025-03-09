import pytest
import numpy as np
from dkmat import *


#################
# asvec
def test_conversion_vecteur_avec_liste():
  result = asvec([0,1,2,3,4,5,6])
  expected = np.array([[0], [1], [2], [3], [4], [5], [6]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


def test_conversion_vecteur():
  result = asvec(0,1,2,3,4,5,6)
  expected = np.array([[0], [1], [2], [3], [4], [5], [6]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


#################
# asmat
def test_conversion_matrice_parametre_r_et_tuple():
  result = asmat((0,1), r=2)
  expected = np.array([[0], [1]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


def test_conversion_matrice_2x2_avec_2_tuples():
  result = asmat((0,1),(0,1))
  expected = np.array([[0,1], [0,1]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


def test_conversion_matrice_2x2_avec_args():
  result = asmat(0,1,2,3)
  expected = np.array([[0, 1], [2, 3]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


def test_conversion_matrice_2x2():
  result = asmat(range(4))
  expected = np.array([[0, 1], [2, 3]])
  # Vérification que la matrice résultante est correcte
  assert np.array_equal(result, expected)


def test_conversion_matrice_2x3_avec_args():
  result = asmat(0,1,2,3,4,5,r=2)
  expected = np.array([[0, 1, 2], [3, 4, 5]])
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
    asmat(range(7), 2)


def test_conversion_matrice_1x1():
  result = asmat([42])
  expected = np.array([[42]])
  assert np.array_equal(result, expected)

#################
# pprint
def test_pprint(capsys):
  pprint(asmat(range(4))/11)
  captured = capsys.readouterr()
  expected_output = " 0   1/11 \n2/11 3/11 \n"
  assert captured.out == expected_output


def test_pprint_sqrt_fraction(capsys):
  pprint(asvec([1 / pow(i,.5) for i in range(1,5)]))
  captured = capsys.readouterr()
  expected_output = " 1  \n √2 \n √3 \n1/2 \n"
  assert captured.out == expected_output


def test_pprint_sqrt(capsys):
  pprint(asvec([pow(i,.5) for i in range(1,5)]))
  captured = capsys.readouterr()
  expected_output = "1  \n√2 \n√3 \n2  \n"
  assert captured.out == expected_output


#################
# rref
def test_rref():
  result = gauss_jordan_rref(asmat(range(4)))
  expected = np.eye(2)
  assert np.array_equal(expected, result)


def test_rref_3x3_avec_nulle():
  result = gauss_jordan_rref(asmat((1,2,3,1,2,3,1,2,4)))
  expected = asmat((1,2,0,0,0,1,0,0,0))
  assert np.array_equal(expected, result)


def test_rref_2x3():
  result = gauss_jordan_rref(asmat(range(6),2))
  result = [[round(i,5) for i in j] for j in result]
  expected = np.hstack((np.eye(2), asmat((-1,2),2)))
  assert np.array_equal(expected, result)
