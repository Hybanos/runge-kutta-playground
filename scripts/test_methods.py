import numpy as np
import sys

import tableau

class Config:
    def __init__(self, start, end, dt):
        self.startt = start
        self.endt = end
        self.dt = dt 

    def __repr__(self):
        return f"[{round(self.startt, 3)};{round(self.endt, 3)}] - dt: {self.dt}"

class ODE:
    def __init__(self, ode, exact, name):
        self.f = ode
        self.exact = exact
        self.name = name

def abstract_rk(ode, tableau, config):
    s = tableau.stages

    t = config.startt
    Y = ode.exact(t)
    dt = config.dt 

    while t < config.endt:
        stages = np.zeros(s)

        stages[0] = ode.f(t + tableau.c[0] * dt, Y)
        for i in range(1, s):
            tmp = 0
            for j in range(i):
                tmp += stages[j] * tableau.A[i][j]
            tmp = Y + tmp * dt
            stages[i] = ode.f(t + tableau.c[i] * dt, tmp)
        
        tmp = 0
        for i in range(s):
            tmp += stages[i] * tableau.b[i]
        Y = Y + tmp * dt

        t += dt

    # return (Y - ode.exact(t)) / abs(ode.exact(t)) 
    return abs(Y - ode.exact(t))


ODEs = [
    # ODE(
    #     lambda x, y: y,
    #     lambda x: np.exp(x),
    #     "y"
    # ),
    # ODE(
    #     lambda x, y: y * y - y,
    #     lambda x: 1 / (1 + np.exp(x)),
    #     "y² - y"
    # ),
    # ODE(
    #     lambda x, y: np.cos(x),
    #     lambda x: np.sin(x),
    #     "cos(x)"
    # ),
    # # divide watch out
    # ODE(
    #     lambda x, y: y * y,
    #     lambda x: 1 / x,
    #     "y²"
    # ),
    # ODE(
    #     lambda x, y: y / x,
    #     lambda x: x,
    #     "y / x"
    # ),
    # ODE(
    #     lambda x, y: np.exp(x) - y,
    #     lambda x: 0.5 * np.exp(x) + np.exp(-x),
    #     "e^x - y"
    # ),
    ODE(
        lambda x, y: -2 * y * y + x * (2 * x + 3) * y - x,
        lambda x:1 / (2 * x + 3),
        "-2y²+x(2x+3)y - x"
    ),
    # ODE(
    #     lambda x, y: np.tan(x) / np.cos(y),
    #     lambda x: np.arcsin(-np.log(np.abs(np.cos(x)))),
    #     "tan(x) / cos(y)"
    # ),
    # ODE(
    #     lambda x, y: -(3 * x * x + 1) * y,
    #     lambda x: 5 * np.exp(-x**3 - x),
    #     "-(3x²+1)y"
    # )
]

if __name__ == "__main__":
    stages = int(sys.argv[1])

    config = Config(0, 1, 0.0001)

    tableaux = tableau.load(stages)
    i = 0
    for o in ODEs:
        for t in tableaux:
            print(f"\r{round(i / (len(ODEs * len(tableaux))) * 100, 3)}%", end="")
            err = abstract_rk(o, t, config)
            t.error = err
            i += 1
    print("")

    tableaux.sort(key=lambda x: -x.error)

    for t in tableaux:
        print(t)