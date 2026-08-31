import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def run(name):
    out = subprocess.run(["./run", "--omp", "-m", name], capture_output=True, env=os.environ.copy().update({"EXEC": "bench"}))

    i = []
    avgs = []
    means = []
    sigmas = []
    reps = []

    for line in out.stdout.decode().split("\n"):
        pattern = r"(\d+) : ([\d\.e-]+)s average, ([\d\.e-]+)s mean, ([\d\.e-]+) sigmas. \((\d+) reps\)"
        result = re.match(pattern, line)
        if (result):
            i.append(int(result.group(1)))
            avgs.append(float(result.group(2)))
            means.append(float(result.group(3)))
            sigmas.append(float(result.group(4)))
            reps.append(int(result.group(5)))

    plt.grid()
    plt.yscale("log")
    plt.xlabel("stages")
    plt.ylabel("time (s)")
    plt.title(f"{name} generation time")
    plt.errorbar(i, means, yerr=sigmas, capsize=3)
    plt.savefig(f"{name}.svg")

run("tree")
run("system")
