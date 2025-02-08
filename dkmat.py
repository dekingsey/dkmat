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

