import random
import numpy as np

def calc_euclidean_distance(cities):
    coordinates = np.array(cities)
    difference = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    euclidean_distance_matrix = np.sqrt((difference ** 2)).sum(axis = 2)
    return euclidean_distance_matrix

def calc_total_distance_travelled(path, euclidean_distance_matrix):
    index = np.array(path + [path[0]])
    return np.sum(euclidean_distance_matrix[index[:-1], index[1:]])

def nearest_neighbor(cities, euclidean_distance_matrix):
    number_of_cities = len(cities)
    unvisited_city = np.ones(number_of_cities, dtype=bool)
    current_visited_cities = np.random.randint(number_of_cities)
    unvisited_city[current_visited_cities] = False
    path = [current_visited_cities]

    for _ in range(number_of_cities - 1):
        distances = euclidean_distance_matrix[current_visited_cities].copy()
        distances[~unvisited_city] = np.inf
        closesst_city = np.argmin(distances)
        path.append(closesst_city)
        unvisited_city[closesst_city] = False
        current_visited_cities = closesst_city
    return path

def generate_population(population_index, cities, euclidean_distance_matrix):
    population = [nearest_neighbor(cities, euclidean_distance_matrix)]
    for _ in range(population_index - 1):
        path = np.random.permutation(len(cities)).tolist()
        population.append(path)
    return population

def ox_crossover(parent1, parent2):
    length = len(parent1)
    child = [None] * length
    initial_state, goal_state = sorted(random.sample(range(length), 2))
    child[initial_state:goal_state+1] = parent1[initial_state:goal_state+1]
    mask = np.isin(parent2, child[initial_state:goal_state+1], invert = True)
    sorted_parent2 = np.array(parent2)[mask]
    filling_index = [i for i, x in enumerate(child) if x is None]
    filling_child = sorted_parent2.tolist()
    for index, city in zip(filling_index, filling_child):
        child[index] = city
    return child

def mutation_operator(path, mutation_ratio):
    path = path.copy()
    for i in range(len(path)):
        if random.random() < mutation_ratio:
            k = random.randint(0, len(path) - 1)
            path[i], path[k] = path[k], path[i]
    return path

def ranking_the_population(population, euclidean_distance_matrix):
    distance = np.array([calc_total_distance_travelled(path, euclidean_distance_matrix) for path in population])
    sorting_index = np.argsort(distance)
    return sorting_index, distance[sorting_index]

def population_fitness(population, elite_index, mutation_ratio, euclidean_distance_matrix):
    ranked_index, ranked_distances = ranking_the_population(population, euclidean_distance_matrix)
    number_of_elites = [population[i] for i in ranked_index[elite_index]]
    mating_top_population = [population[i] for i in ranked_index[:len(population)//2]]
    children = number_of_elites.copy()
    top_population_size = len(mating_top_population)
    parent_index = np.random.randint(0, top_population_size, size=(len(population) - elite_index, 2))

    for index1, index2 in parent_index:
        parent1 = mating_top_population[index1]
        parent2 = mating_top_population[index2]
        child = ox_crossover(parent1, parent2)
        child = mutation_operator(child, mutation_ratio)
        children.append(child)
    return children

def reading_input_files(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    N = int(lines[0])
    cities = [tuple(map(int, line.strip().split())) for line in lines[1:N+1]]
    return cities


def writing_output_files(filename, path, distance, cities):
    with open(filename, 'w') as file:
        file.write(f"{distance:.3f}\n")
        for city_index in path + [path[0]]:
            city = cities[city_index]
            file.write(f"{city[0]} {city[1]} {city[2]}\n")

def two_opt_local_search(path, euclidean_distance_matrix, total_iteration=50):
    shortest_distance = calc_total_distance_travelled(path, euclidean_distance_matrix)
    optimal_path = path.copy()
    better_path = True
    count = 0
    while better_path and count < total_iteration:
        better_path = False
        for i in range(1, len(optimal_path) - 2):
            for k in range(i + 1, len(optimal_path)):
                if k - i == 1:
                    continue
                novel_path = optimal_path[:i] + optimal_path[i:k+1][::-1] + optimal_path[k+1:]
                novel_distance = calc_total_distance_travelled(novel_path, euclidean_distance_matrix)
                if novel_distance < shortest_distance:
                    optimal_path = novel_path
                    shortest_distance = novel_distance
                    better_path = True
                    break
            if better_path:
                break
        count += 1
    return optimal_path

def tsp_genetic_algorithm(input_file, output_file, population_index=100, generations=150, elite_index=10, mutation_ratio=0.02):
    cities = reading_input_files(input_file)
    euclidean_distance_matrix = calc_euclidean_distance(cities)
    population = generate_population(population_index, cities, euclidean_distance_matrix)
    
    for _ in range(generations):
        population = population_fitness(population, elite_index, mutation_ratio, euclidean_distance_matrix)
    
    ranked_index, ranked_distances = ranking_the_population(population, euclidean_distance_matrix)
    best_path_index = ranked_index[0]
    optimal_path = population[best_path_index]
    optimal_path = two_opt_local_search(optimal_path, euclidean_distance_matrix, total_iteration=50)
    shortest_distance = calc_total_distance_travelled(optimal_path, euclidean_distance_matrix)
    writing_output_files(output_file, optimal_path, shortest_distance, cities)

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    tsp_genetic_algorithm(input_file, output_file)