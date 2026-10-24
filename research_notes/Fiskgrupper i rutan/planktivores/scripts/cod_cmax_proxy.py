"""Hansson et al. 1996 North Sea cod Cmax (Kitchell / CEQ 2 temperature function)
evaluated at Norway pout (25 g) and blue whiting (100 g) body sizes.
Cmax = a W^b F(T), g wet prey / g wet fish / d. Parameters: Hansson et al. 1996 Table 2."""
import math
A, B, Q, TOPT, TMAX = 0.099, -0.195, 2.41, 13.7, 21.0

def f_t(t):
    v = (TMAX - t) / (TMAX - TOPT)
    if v <= 0:
        return 0.0
    z = math.log(Q) * (TMAX - TOPT)
    y = math.log(Q) * (TMAX - TOPT + 2)
    x = (z ** 2 * (1 + (1 + 40 / y) ** 0.5) ** 2) / 400
    return v ** x * math.exp(x * (1 - v))

def cmax(w, t):
    return A * w ** B * f_t(t)

if __name__ == "__main__":
    t50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]
    print("T  F(T)  Cmax%BW/d: 10g 25g 50g 100g 200g")
    for t in [4, 6, 7, 8, 10, 12, 13.7]:
        print(t, round(f_t(t), 3), [round(100 * cmax(w, t), 2) for w in (10, 25, 50, 100, 200)])
    print("50 m monthly, 25 g / 100 g:")
    for m, t in enumerate(t50, 1):
        print(m, t, round(100 * cmax(25, t), 2), round(100 * cmax(100, t), 2))
    mean = lambda w: sum(cmax(w, t) for t in t50) / 12
    print("annual mean 50 m: 25 g %.2f, 100 g %.2f %%BW/d" % (100 * mean(25), 100 * mean(100)))
    # apparent Q10 of F(T) between 6 and 8 C
    print("Q10 of Cmax over 6-8 C: %.2f" % ((f_t(8) / f_t(6)) ** (10 / 2)))

def f_kitchell(t, q, topt, tmax):
    v = (tmax - t) / (tmax - topt)
    if v <= 0:
        return 0.0
    z = math.log(q) * (tmax - topt)
    y = math.log(q) * (tmax - topt + 2)
    x = (z ** 2 * (1 + (1 + 40 / y) ** 0.5) ** 2) / 400
    return v ** x * math.exp(x * (1 - v))

if __name__ == "__main__":
    print("near-bottom flat 6/7/8 C, 25 g:", [round(100 * cmax(25, t), 2) for t in (6, 7, 8)],
          "100 g:", [round(100 * cmax(100, t), 2) for t in (6, 7, 8)])
    # Hansson 1996 respiration temperature function: Q 1.88, Topt 21, Tmax 24
    r = lambda t: f_kitchell(t, 1.88, 21.0, 24.0)
    for lo, hi in ((6, 8), (6, 10), (7, 12), (4, 14)):
        print("cod respiration F(T) apparent Q10 %g-%g C: %.2f" % (lo, hi, (r(hi) / r(lo)) ** (10 / (hi - lo))))
