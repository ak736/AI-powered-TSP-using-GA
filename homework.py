import random
import numpy as np

def compute_distance_matrix(cities):
    coords = np.array(cities)
    delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance_matrix = np.sqrt((delta ** 2).sum(axis=2))
    return distance_matrix

def calculate_total_distance(path, distance_matrix):
    idx = np.array(path + [path[0]])
    return np.sum(distance_matrix[idx[:-1], idx[1:]])

def greedy_nearest_neighbor(cities, distance_matrix):
    num_cities = len(cities)
    unvisited = np.ones(num_cities, dtype=bool)
    current_city = np.random.randint(num_cities)
    unvisited[current_city] = False
    path = [current_city]
    
    for _ in range(num_cities - 1):
        distances = distance_matrix[current_city].copy()
        distances[~unvisited] = np.inf
        nearest_city = np.argmin(distances)
        path.append(nearest_city)
        unvisited[nearest_city] = False
        current_city = nearest_city
    
    return path

def create_initial_population(population_size, cities, distance_matrix):
    population = [greedy_nearest_neighbor(cities, distance_matrix)]
    for _ in range(population_size - 1):
        path = np.random.permutation(len(cities)).tolist()
        population.append(path)
    return population

def order_crossover(parent1, parent2):
    length = len(parent1)
    child = [None] * length
    start, end = sorted(random.sample(range(length), 2))
    child[start:end+1] = parent1[start:end+1]
    mask = np.isin(parent2, child[start:end+1], invert=True)
    filtered_parent2 = np.array(parent2)[mask]
    fill_indices = [i for i, x in enumerate(child) if x is None]
    child_fill = filtered_parent2.tolist()
    for idx, city in zip(fill_indices, child_fill):
        child[idx] = city
    return child

def mutate(path, mutation_rate):
    path = path.copy()
    for i in range(len(path)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(path) - 1)
            path[i], path[j] = path[j], path[i]
    return path

def rank_population(population, distance_matrix):
    distances = np.array([calculate_total_distance(path, distance_matrix) for path in population])
    sorted_indices = np.argsort(distances)
    return sorted_indices, distances[sorted_indices]

def evolve_population(population, elite_size, mutation_rate, distance_matrix):
    ranked_indices, ranked_distances = rank_population(population, distance_matrix)
    elites = [population[i] for i in ranked_indices[:elite_size]]
    mating_pool = [population[i] for i in ranked_indices[:len(population)//2]]
    children = elites.copy()
    pool_size = len(mating_pool)
    parent_indices = np.random.randint(0, pool_size, size=(len(population) - elite_size, 2))
    
    for idx1, idx2 in parent_indices:
        parent1 = mating_pool[idx1]
        parent2 = mating_pool[idx2]
        child = order_crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        children.append(child)
    
    return children

def read_input_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    N = int(lines[0])
    cities = [tuple(map(int, line.strip().split())) for line in lines[1:N+1]]
    return cities

def write_output_file(filename, path, distance, cities):
    with open(filename, 'w') as file:
        file.write(f"{distance:.3f}\n")
        for city_index in path + [path[0]]:
            city = cities[city_index]
            file.write(f"{city[0]} {city[1]} {city[2]}\n")

def two_opt(path, distance_matrix, max_iter=50):
    best_distance = calculate_total_distance(path, distance_matrix)
    best_path = path.copy()
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        for i in range(1, len(best_path) - 2):
            for k in range(i + 1, len(best_path)):
                if k - i == 1:
                    continue
                new_path = best_path[:i] + best_path[i:k+1][::-1] + best_path[k+1:]
                new_distance = calculate_total_distance(new_path, distance_matrix)
                if new_distance < best_distance:
                    best_path = new_path
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        iter_count += 1
    return best_path

def tsp_genetic_algorithm(input_file, output_file, population_size=100, generations=150, elite_size=10, mutation_rate=0.02):
    cities = read_input_file(input_file)
    distance_matrix = compute_distance_matrix(cities)
    population = create_initial_population(population_size, cities, distance_matrix)
    
    for _ in range(generations):
        population = evolve_population(population, elite_size, mutation_rate, distance_matrix)
    
    ranked_indices, ranked_distances = rank_population(population, distance_matrix)
    best_path_index = ranked_indices[0]
    best_path = population[best_path_index]
    best_path = two_opt(best_path, distance_matrix, max_iter=50)
    best_distance = calculate_total_distance(best_path, distance_matrix)
    write_output_file(output_file, best_path, best_distance, cities)

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    tsp_genetic_algorithm(input_file, output_file)
