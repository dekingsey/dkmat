import numpy as np
import numpy.linalg as la

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

if __name__ == "__main__":
  print("Attendu: matrice 2x2 contenant les nombres de 0 à 3")
  print(asmat(range(4)))
  print("Attendu: matrice 2x3 contenant les nombres de 0 à 5")
  print(asmat(range(6),2))
  print("Attendu: matrice 3x3 contenant les nombres de 0 à 8")
  print(asmat(range(9)))
  print("Attendu: exception")
  print(asmat(range(7)))
