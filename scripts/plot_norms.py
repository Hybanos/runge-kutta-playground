import matplotlib.pyplot as plt
import subprocess

options = [
    ["--no-backtrack", "--no-levenberg"],
    ["--no-levenberg"],
    ["--no-backtrack"],
    []
]

names = [
    "none",
    "backtrack",
    "levengerg",
    "both"
]

for name, opt in zip(names, options):

    args = ["./run", "--cuda", "-s10", "-n1", "-i400", "--wipe", "--seed", "50"] + opt
    print(args)
    out = subprocess.run(args, capture_output=True)

    savenext = False
    norms = []
    for line in out.stdout.decode().split("\n"):
        if savenext:
            norms.append(float(line.strip()))
        if line.startswith("matrix: norms"):
            savenext = True
        else:
            savenext = False

    plt.plot(norms, label=name)

plt.grid()
plt.yscale("log")
plt.ylabel("residual norm")
plt.xlabel("iterations")
plt.ylim(top=1e5, bottom=5e-3)
plt.title("s=10 norms")
plt.legend(loc="upper right")
plt.show()