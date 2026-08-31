import matplotlib.pyplot as plt

files = ["none.txt", "backtrack.txt", "levenberg.txt", "both.txt"]
for fname in files:
    with open(fname, "r") as f:
        data = f.readlines()

    savenext = False
    norms = []
    for line in data:
        if savenext:
            norms.append(float(line.strip()))
        if line.startswith("matrix: norms"):
            savenext = True
        else:
            savenext = False

    plt.plot(norms, label=fname.split(".")[0])

plt.grid()
plt.yscale("log")
plt.ylabel("residual norm")
plt.xlabel("iterations")
plt.ylim(top=1e5, bottom=1e-2)
plt.title("s=10 norms")
plt.legend(loc="upper right")
plt.show()