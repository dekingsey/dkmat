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

if __name__ == "__main__":
  pprint(asmat(range(4))/11)
