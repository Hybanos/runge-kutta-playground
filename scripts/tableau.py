import json
import numpy as np

class Tableau:
    def __init__(self, stages, _json):
        self.stages = stages
        self.loss = _json["loss"]
        self.A = np.array(_json["a"])
        self.b = np.array(_json["b"])
        self.c = np.array(_json["c"])

        self.error = 0.0

    def __repr__(self):
        return f"{self.stages}: {self.loss},\terr: {self.error}\na: {self.A}\nb: {self.b}\n c: {self.c}"

def load(stages):
    out = []
    path = f"./cache/tableaux/solutions/s{stages}.json"
    with open(path, "r") as f:
        j = json.load(f)
    for o in j:
        out.append(Tableau(stages, o))
    return out

if __name__ =="__main__":
    tableaux = load(10)
    for t in tableaux:
        print(t)
        print(t.a)
