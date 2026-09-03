import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def bech_exec():
    i = []
    avgs = []
    means = []
    sigmas = []
    reps = []

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

        plt.errorbar(i, means, yerr=sigmas, capsize=3, ecolor="gray", label=name)
        plt.xticks(i)



    run("system")
    run("tree")

    plt.grid()
    plt.yscale("log")
    plt.xlabel("stages")
    plt.ylabel("time (s)")
    plt.title(f"Trees and Systems generation time")
    plt.legend()
    plt.savefig(f"out.svg")

def bench_perf():
    cores = [4, 8, 16, 32, 64, 128, 256]
    iters = 10

    if not os.path.exists("perf_out.bin"):
        data = np.zeros((len(cores), iters))
        for c in cores:
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": str(c)
            })
            cmd = [
                "./run",
                "--omp",
                "-s10",
                "-n1000",
                f"-i{iters}",
                "--wipe",
                "--checkpoint-save-freq", "10000",
                "--seed", "100"
            ]
            out = subprocess.run(cmd, capture_output=True, env=env)
            print(out.stderr.decode())
            i = 0
            for line in out.stdout.decode().split("\n"):
                pattern = r"\d+ ips: (\d+), ([\d\.]+)s"
                match = re.match(pattern, line)
                if match is not None:
                    data[cores, i] = match.group(2)
                    i += 1
            print(c, data)

        data.tofile("perf_out.bin")

    data = np.fromfile("perf_out.bin")
    print(data)

def plot_mem():
    def format_size(mb):
        if mb < 1:
            value, unit = mb * 1024, "KB"
        elif mb < 1024:
            value, unit = mb, "MB"
        else:
            value, unit = mb / 1024, "GB"

        return f"{value:.3g} {unit}"
    n_range = list(range(50, 1000, 50))
    s_range = list(range(2, 11))
    if not os.path.exists("ram_cost.bin"):
        data = np.zeros((len(n_range), len(s_range)))
        for i_n, n in enumerate(n_range):
            for i_s, s in enumerate(s_range):
                cmd = [
                    "./run",
                    "--omp",
                    f"-s{s}",
                    f"-n{n}",
                    "-i0",
                    "--dump-mem",
                    "--wipe",
                ]
                out = subprocess.run(cmd, capture_output=True)
                print(out.stdout.decode())
                print(out.stderr.decode())
                total_size = 0
                for line in out.stdout.decode().split("\n"):
                    pattern = r"^.+ (\d+)$"
                    match = re.match(pattern, line)
                    if match is not None:
                        total_size += int(match.group(1))
                    
                data[i_n, i_s] = total_size

        data.tofile("ram_cost.bin")

    data = np.fromfile("ram_cost.bin").reshape(len(n_range), len(s_range))
    data = data / 1000000
    data = data[::-1, :]

    ax = plt.subplot(111)

    ax.imshow(data, norm="log", cmap="coolwarm", aspect="0.5")
    ax.set_xlabel("stage count")
    ax.set_ylabel("parallel instances")
    ax.set_xticks(range(len(s_range)), labels=s_range)
    ax.set_yticks(range(len(n_range)), labels=n_range[::-1])

    for (j, i), label in np.ndenumerate(data):
        plt.text(i, j, format_size(label), ha="center", va="center")

    plt.gcf().set_size_inches(10, 10)
    plt.title("Memory consumption")
    plt.savefig("memory.svg")
    plt.show()

if __name__ == "__main__":
    # plot_mem()
    bench_perf()