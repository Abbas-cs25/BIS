
import numpy as np
import matplotlib.pyplot as plt

distance = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

coordinates = np.array([
    [2, 8],
    [8, 9],
    [9, 2],
    [2, 1]
])

n = len(distance)
pheromone = np.ones((n, n))
best_route = None
best_distance = float("inf")

for iteration in range(10):
    for ant in range(4):
        route = [ant]
        unvisited = list(range(n))
        unvisited.remove(ant)

        while unvisited:
            current = route[-1]
            next_city = min(
                unvisited,
                key=lambda city: distance[current][city]
            )
            route.append(next_city)
            unvisited.remove(next_city)

        total = sum(
            distance[route[i]][route[i + 1]]
            for i in range(n - 1)
        )

        total += distance[route[-1]][route[0]]

        if total < best_distance:
            best_distance = total
            best_route = route.copy()

    pheromone *= 0.8

    for i in range(n):
        pheromone[best_route[i]][best_route[(i + 1) % n]] += 1 / best_distance

    print(
        "Iteration", iteration + 1,
        "Best Distance:", best_distance
    )

best_route = best_route + [best_route[0]]

print("Best Route:", best_route)
print("Shortest Distance:", best_distance)

plt.figure(figsize=(8, 6))

for i in range(n):
    plt.scatter(
        coordinates[i][0],
        coordinates[i][1],
        color="red",
        s=150
    )

    plt.text(
        coordinates[i][0] + 0.2,
        coordinates[i][1] + 0.2,
        f"City {i}",
        fontsize=12
    )

for i in range(len(best_route) - 1):
    city1 = best_route[i]
    city2 = best_route[i + 1]

    x = [
        coordinates[city1][0],
        coordinates[city2][0]
    ]

    y = [
        coordinates[city1][1],
        coordinates[city2][1]
    ]

    plt.plot(x, y, "b-", linewidth=2)

plt.scatter(
    coordinates[best_route[0]][0],
    coordinates[best_route[0]][1],
    color="green",
    s=200,
    label="Starting City"
)

plt.title("Ant Colony Optimization - TSP")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.show()
