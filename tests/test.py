import sympy
import numpy as np

x = 1 / (np.arange(1, 7, 1))
# x = np.random.random(6) * 4 - 2
print("x:\n", x.T)

symbols = sympy.symbols("b1 b2 b3 c2 c3 a32")
b1 = symbols[0]
b2 = symbols[1]
b3 = symbols[2]
c2 = symbols[3]
c3 = symbols[4]
a32 = symbols[5]

f = [
    b1 + b2 + b3 - 1,
    b2 * c2 + b3 * c3 - 1/2,
    b2 * c2 * c2 + b3 * c3 * c3 - 1/3,
    b3 * a32 * c2 - 1/6
]

J = []
for e in f:
    tmp = []
    for s in symbols:
        tmp.append(sympy.diff(e, s))
    J.append(tmp)

print(J)

lf = sympy.lambdify(symbols, f)
lJ = sympy.lambdify(symbols, J)

while True:
    try:
        for i in range(1):
            ff = np.array(lf(*x))
            JJ = np.array(lJ(*x))

            print("f:\n", ff.T)
            print("J:\n", JJ.T)

            A = JJ.T @ JJ
            b = -JJ.T @ ff

            print("A:\n", A.T)
            print("det A:", np.linalg.det(A))
            print("b:\n", b.T)

            dx = np.linalg.solve(A, b)
            print("dx:\n", dx.T)

            x += dx
        exit(0)
    except KeyboardInterrupt as e:
        exit(0)
    except Exception as e:
        x = np.random.random(6) * 4 - 2
        print("\n\n\n\n\n")
        print("x:\n", x)
        exit()

