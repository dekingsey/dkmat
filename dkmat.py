import numpy as np
import numpy.linalg as la
from fractions import Fraction

def asvec(*args):
  """
  Convertit les paramètres reçus en un vecteur.
  :param args: Tous les nombres composant le vecteur.
  :return Un vecteur sous forme matricielle, c'est à dire une matrice Nx1. 
  """
  v=asmat(args, 1)
  return(v.T)

def asmat(*args, **kwargs):
  """
  Convertit les paramètres reçus en une matrice.

  Exemples d'utilisation:
   - asmat([1,2,3,4]) retournera une matrice 2x2 (on utiliser la racine carrée pour déterminer le nombre de rangées)
   - asmat((1,2,3,4,5,6), 2) retournera une matrice 2x3
   - asmat(1,2,3,4,5,6, r=2) retournera une matrice 2x3
   - asmat(1,2,3,4) retournera une matrice 2x2 (on utiliser la racine carrée pour déterminer le nombre de rangées)
  :param args: Tous les nombres composants le vecteur ou encore une énumération suivi d'une nombre de lignes.
  :param kwargs: 'r' peut être spécifié pour forcer un nombre de lignes. Sinon, on aura une matrice carrée avec un nombre de colonnes plus grand que le nombre de rangées avec un nombre de rangées le plus près possible de la racine carrée.
  :return Une matrice formée des nombres reçus.
  """
  r = 0

  if "r" in kwargs:
    r = kwargs["r"]

  # twiste de la muerte ici: je convertis args[0] en liste, si c'est déjà une liste, un tuple ou un array
  # ndim va me retourner sa dimension + 1, sinon, cela va retourner 1.
  if (len(args) == 2 and np.ndim([args[0]]) > 1 and isinstance(args[1], int) and r==0):
    l = args[0]
    r = args[1]
  # encore la même twiste de la muerte ici
  elif (len(args) == 1 and np.ndim([args[0]]) > 1):
    l = args[0]
  else:
    l = args
    
  arr = np.asarray(l)
  # aplatir arr pour que len() fonctionne bien - on veut une matrice en output peu importe le nombre de dimensions à l'entrée, tsé
  arr = np.reshape(arr, (-1))
  lenl = len(arr)

  if (r==0):
    r = int(pow(lenl, .5))
    # si possible, on utilise la racine carrée (matrice carrée)
    # sinon, on aura une matrice carrée avec un nombre de colonnes plus grand que le nombre de rangées
    # avec un nombre de rangées le plus près possible de la racine carrée
    while (lenl % r != 0):
      r -= 1

  c = lenl//r

  if (c * r != lenl):
    raise ValueError(f"Une liste de longueur {lenl} ne peut être convertie en matrice {r}x{c}")

  arr = arr.reshape(r, lenl//r)

  return(arr)


def jolie_fraction(n):
  """
  Transforme un nombre en format fractionnel. Si les termes de la fraction sont des racines carrées, ils seront affichés avec l'opérateur √.
  :param n: Nombre à transformer.
  :return Une chaîne de caractères x/y.
  """
  LIMITE = 10E3
  f = Fraction(n).limit_denominator()
  if f.numerator < LIMITE and f.denominator < LIMITE:
    return str(f)
  else:
    f2 = Fraction(pow(n, 2)).limit_denominator()
    if f2.numerator < LIMITE and f2.denominator < LIMITE:
      retval = []
      for i in f2.as_integer_ratio():
        carre = pow(i,.5)
        if carre - int(carre) < 1/LIMITE:
          # si on est au denominateur, retval existe
          # si le denominateur est 1, on ignore le denominateur
          if retval and carre != 1:
            retval.append(str(int(carre)))
        else:
          retval.append(f"√{i}")
      return "/".join(retval)
    else:
      return str(f)


def pprint(A):
  """
  Affiche sous forme d'un tableau une matrice reçue en paramètre. Les nombres rationnels seront affichés sous forme de fraction. Les racines carrées seront affichées avec l'opérateur √.
  :param A: Matrice à afficher.
  """
  if isinstance(A, np.ndarray) and A.ndim == 2:
    l = []
    maxlen = 0
    for r in A:
      rl = []
      l.append(rl)
      for v in r:
        sv = jolie_fraction(v)
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
  """
  Réduit une matrice sous sa forme échelonnée.
  :param A: Matrice à réduire.
  :return: Matrice réduite.
  """
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
