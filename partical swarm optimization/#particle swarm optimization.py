#particle swarm optimization
import numpy as np
import matplotlib.pyplot as plt

local_target = np.array([3, 3])
global_target = np.array([8, 8])

def temperature(position):
    local_heat = 20 * np.exp(-np.sum((position - local_target)**2))
    global_heat = 40 * np.exp(-np.sum((position - global_target)**2))
    return local_heat + global_heat

np.random.seed(10)
robots = np.random.uniform(0, 10, (3, 2))

best_position = robots[0].copy()
best_temperature = temperature(best_position)

paths = [robots.copy()]

for step in range(10):
    for i in range(3):
        current_temp = temperature(robots[i])

        if current_temp > best_temperature:
            best_temperature = current_temp
            best_position = robots[i].copy()

        direction = best_position - robots[i]
        robots[i] += 0.3 * direction + np.random.uniform(-0.5, 0.5, 2)
        robots[i] = np.clip(robots[i], 0, 10)

    paths.append(robots.copy())
    print(f"Iteration {step + 1}: Best temperature = {best_temperature:.4f}")

print("\nBest location found:", best_position)
print("Best temperature:", best_temperature)

paths = np.array(paths)

for i in range(3):
    plt.plot(paths[:, i, 0], paths[:, i, 1], marker="o",
             label=f"Robot {i+1}")

plt.scatter(*local_target, color="orange", s=150, label="Local maximum")
plt.scatter(*global_target, color="red", s=150, label="Global maximum")

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("PSO Robotic Search - 10 Iterations")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()
plt.legend()
plt.show()
