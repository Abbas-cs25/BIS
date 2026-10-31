#cuckoo search

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

a = np.array([100, 120, 150], dtype=float)
b = np.array([5.0, 5.5, 5.0], dtype=float)
c = np.array([0.004, 0.006, 0.009], dtype=float)

Pmin = np.array([50, 50, 50], dtype=float)
Pmax = np.array([200, 150, 100], dtype=float)

PD = 300.0
N = len(a)

def fuel_cost(P):
    return np.sum(a + b * P + c * P**2)

def repair_solution(P):
    P = np.clip(P, Pmin, Pmax)

    lower = -10000.0
    upper = 10000.0

    for _ in range(100):
        mid = (lower + upper) / 2
        corrected = np.clip(P - mid, Pmin, Pmax)

        if np.sum(corrected) > PD:
            lower = mid
        else:
            upper = mid

    return np.clip(
        P - (lower + upper) / 2,
        Pmin,
        Pmax
    )

def levy_flight(size, beta=1.5):
    sigma = (
        math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        /
        (
            math.gamma((1 + beta) / 2)
            * beta
            * 2**((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)

    return u / (np.abs(v)**(1 / beta) + 1e-12)

def modified_cuckoo_search(
    population=30,
    iterations=500,
    pa=0.25
):

    nests = np.random.uniform(
        Pmin,
        Pmax,
        (population, N)
    )

    nests = np.array([
        repair_solution(x) for x in nests
    ])

    fitness = np.array([
        fuel_cost(x) for x in nests
    ])

    best_index = np.argmin(fitness)

    best_solution = nests[best_index].copy()
    best_cost = fitness[best_index]

    convergence = []

    for t in range(iterations):

        alpha = 0.5 * (1 - t / iterations) + 0.01

        for i in range(population):

            step = levy_flight(N)

            candidate = (
                nests[i]
                + alpha * step * (nests[i] - best_solution)
                + alpha * step * (Pmax - Pmin)
            )

            candidate = repair_solution(candidate)

            candidate_cost = fuel_cost(candidate)

            j = np.random.randint(population)

            if candidate_cost < fitness[j]:
                nests[j] = candidate.copy()
                fitness[j] = candidate_cost

        number_abandoned = max(
            1,
            int(pa * population)
        )

        worst_indices = np.argsort(fitness)[
            -number_abandoned:
        ]

        for j in worst_indices:

            random_solution = np.random.uniform(
                Pmin,
                Pmax,
                N
            )

            if np.random.rand() < 0.5:
                random_solution = (
                    best_solution
                    + np.random.normal(0, 0.1, N)
                    * (Pmax - Pmin)
                )

            nests[j] = repair_solution(random_solution)
            fitness[j] = fuel_cost(nests[j])

        current_best = np.argmin(fitness)

        if fitness[current_best] < best_cost:
            best_cost = fitness[current_best]
            best_solution = nests[current_best].copy()

        convergence.append(best_cost)

    return best_solution, best_cost, convergence

np.random.seed(42)

best_P, minimum_cost, convergence = modified_cuckoo_search(
    population=30,
    iterations=500,
    pa=0.25
)

results = pd.DataFrame({
    "Generator": [f"G{i+1}" for i in range(N)],
    "Power Output (MW)": best_P,
    "Fuel Cost ($/hr)": a + b * best_P + c * best_P**2
})

print("\nECONOMIC LOAD DISPATCH RESULTS")
print(results.to_string(index=False))

print("\nTotal Power Generation:",
      round(np.sum(best_P), 4), "MW")

print("Load Demand:", PD, "MW")

print("Power Balance Error:",
      abs(np.sum(best_P) - PD), "MW")

print("Minimum Fuel Cost:",
      round(minimum_cost, 4), "$/hr")

plt.figure(figsize=(9, 5))
plt.plot(convergence, color="blue", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Total Fuel Cost ($/hr)")
plt.title("Modified Cuckoo Search Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(
    results["Generator"],
    results["Power Output (MW)"],
    color="green"
)
plt.xlabel("Generating Units")
plt.ylabel("Power Output (MW)")
plt.title("Optimal Power Generation Schedule")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
genetic algoritm for optimization problem