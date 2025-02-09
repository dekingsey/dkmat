import numpy as np
import numpy.linalg as la
from fractions import Fraction

def asmat(l, r=0):
  arr = np.asarray(l)
  lenl = len(l)

  if (r==0):
    r = int(pow(lenl, .5))

  c = lenl//r

  if (c * r != lenl):
    raise ValueError(f"Une liste de longueur {lenl} ne peut être convertie en matrice {r}x{c}")

  arr = arr.reshape(r, lenl//r)

  return(arr)

def pprint(A):
  if isinstance(A, np.ndarray) and A.ndim == 2:
    l = []
    maxlen = 0
    for r in A:
      rl = []
      l.append(rl)
      for v in r:
        sv = str(Fraction(v).limit_denominator())
        rl.append(sv)
        maxlen = max(len(sv), maxlen)
    for rl in l:
      sr = ""
      for sv in rl:
        sr += sv.center(maxlen) + " "  
      print(sr)
  else:
    print(A)


def gauss_jordan_rref(A):
    # Copie de la matrice pour ne pas modifier l'originale
    A = A.astype(float)  # Utiliser un type float pour éviter les erreurs lors de la division
    rows, cols = A.shape
    row = 0

    for col in range(cols):
        # Chercher la ligne avec un pivot non nul
        if np.any(A[row: , col] != 0):  # Vérifier s'il existe un pivot non nul dans la colonne
            # Chercher la ligne avec un pivot non nul dans cette colonne (pivot au dessus de la ligne actuelle)
            for i in range(row, rows):
                if A[i, col] != 0:
                    # Echanger les lignes pour amener le pivot à la position de la ligne actuelle
                    A[[row, i]] = A[[i, row]]
                    break

            # Normaliser la ligne du pivot
            A[row] = A[row] / A[row, col]
            
            # Annuler les éléments sous et au-dessus du pivot
            for i in range(rows):
                if i != row:
                    A[i] = A[i] - A[i, col] * A[row]
            row += 1

        # Si tous les pivots sont trouvés, on arrête la réduction
        if row == rows:
            break

    return A
