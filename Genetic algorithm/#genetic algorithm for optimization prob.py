#genetic algorithm for optimization problem

import numpy as np
import matplotlib.pyplot as plt
import random

START = np.array([0, 0])
GOAL = np.array([10, 10])

obstacles = [
    (2, 2, 4, 7),
    (5, 5, 7, 9),
    (7, 1, 9, 4)
]

POPULATION_SIZE = 50
GENERATIONS = 150
MUTATION_RATE = 0.2
NUM_WAYPOINTS = 6

def create_path():
    return np.random.uniform(0, 10, (NUM_WAYPOINTS, 2))

def point_collision(point):
    x, y = point

    for xmin, ymin, xmax, ymax in obstacles:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True

    return False

def segment_collision(p1, p2):
    for t in np.linspace(0, 1, 30):
        point = p1 + t * (p2 - p1)

        if point_collision(point):
            return True

    return False

def fitness(path):
    complete_path = np.vstack([START, path, GOAL])

    distance = 0
    collision_penalty = 0

    for i in range(len(complete_path) - 1):
        p1 = complete_path[i]
        p2 = complete_path[i + 1]

        distance += np.linalg.norm(p2 - p1)

        if segment_collision(p1, p2):
            collision_penalty += 100

    return distance + collision_penalty

population = [
    create_path()
    for _ in range(POPULATION_SIZE)
]

def selection(population):
    candidates = random.sample(population, 3)
    return min(candidates, key=fitness)

def crossover(parent1, parent2):
    child = np.copy(parent1)
    point = random.randint(1, NUM_WAYPOINTS - 1)
    child[point:] = parent2[point:]
    return child

def mutation(path):
    new_path = np.copy(path)

    for i in range(NUM_WAYPOINTS):
        if random.random() < MUTATION_RATE:
            new_path[i] += np.random.normal(0, 1.0, 2)
            new_path[i] = np.clip(new_path[i], 0, 10)

    return new_path

best_path = None
best_cost = float("inf")

for generation in range(GENERATIONS):
    new_population = []

    population.sort(key=fitness)

    new_population.append(population[0].copy())

    while len(new_population) < POPULATION_SIZE:
        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)
        child = mutation(child)

        new_population.append(child)

    population = new_population

    current_best = min(population, key=fitness)
    current_cost = fitness(current_best)

    if current_cost < best_cost:
        best_cost = current_cost
        best_path = current_best.copy()

    if generation % 10 == 0:
        print("Generation:", generation,
              "Best Cost:", round(best_cost, 2))

complete_path = np.vstack([START, best_path, GOAL])

plt.figure(figsize=(8, 8))

for xmin, ymin, xmax, ymax in obstacles:
    plt.gca().add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            color="red",
            alpha=0.5
        )
    )

plt.plot(
    complete_path[:, 0],
    complete_path[:, 1],
    "b-o",
    linewidth=2,
    label="Robot Path"
)

plt.scatter(START[0], START[1],
            color="green", s=150, label="Start")

plt.scatter(GOAL[0], GOAL[1],
            color="blue", s=150, label="Goal")

plt.xlim(0, 10)
plt.ylim(0, 10)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Robot Motion Planning Using Genetic Algorithm")

plt.grid(True)
plt.legend()
plt.show()

print("Best Path Cost:", best_cost)
print("Best Path:")
print(complete_path)