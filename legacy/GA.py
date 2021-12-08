# ==================================== |
# Name		    : 	Dalon Francis Lobo |
# Student ID	: 	202006328          |
# Email		    :  	x2020fyh@stfx.ca   |
# ==================================== |
import numpy as np
from pathlib import Path


def get_portable_path() -> Path:
    """Utility for getting a sensible working directory whether running as a script or in Colab"""
    try:
        outdir = Path(__file__).resolve().parent
        return outdir
    except NameError:
        print("Possible use of Colab detected. Attempting to exploit `globals()`...")
    try:
        outdir = Path(globals()["_dh"][0]).resolve()
        return outdir
    except KeyError:
        print("Colab not detected.")
        print("Defaulting to current working directory for files.")
        return Path().resolve()


class GenticAlgorithm:
    def __init__(
        self,
        max_gen: int,
        pop_size: int,
        cross_rate: float,
        mut_rate: float,
        tour_size: int,
        data_dir: str = None,
        dist_matrix: np.ndarray = None,
        elite_size: int = 10,
        **kwargs,
    ) -> None:
        # Load the distance bw cities in Ajacency matrix
        if data_path:
            try:
                self.distance_matrix = np.loadtxt(data_path)
            except FileExistsError:
                print(f"Please make sure the file exists in the path: {data_path}")
        elif dist_matrix is not None:
            self.distance_matrix = dist_matrix
        else:
            raise Exception("No data available")
        self.population = []
        self.chromozome_len = self.distance_matrix.shape[0]
        self.max_generation = max_gen
        self.population_size = pop_size
        self.fitness_scores = [0] * self.population_size
        self.crossover_rate = cross_rate
        self.mutation_rate = mut_rate
        self.tournament_size = tour_size
        self.elite_size = elite_size

    def init_population(self, population_size: int, chromozome_len: int) -> list:
        """Generate the initial population"""
        return [
            np.random.choice(chromozome_len, chromozome_len, replace=False)
            for i in range(population_size)
        ]

    def get_total_distance(self, chromozome: list) -> float:
        """Calculate the total distance of given chromozome"""
        distance = 0
        # get the distance from ajacency matrix
        for i in range(1, len(chromozome)):
            distance += self.distance_matrix[chromozome[i - 1]][chromozome[i]]
        # distance to reach back to the starting city
        distance += self.distance_matrix[chromozome[-1]][chromozome[0]]
        return distance

    def get_population_fitness(self) -> None:
        """Calculate the fittness for the entire population. Fitness score is inverse of distance."""
        for idx, chromozome in enumerate(self.population):
            self.fitness_scores[idx] = 1 / self.get_total_distance(chromozome)

    def tournament_selection(self) -> np.ndarray:
        """Implement tournament slection stratergy"""
        best_idx, best_score = -1, -1
        for chromo_idx in np.random.choice(self.population_size, self.tournament_size):
            if self.fitness_scores[chromo_idx] > best_score:
                best_score = self.fitness_scores[chromo_idx]
                best_idx = chromo_idx
        return np.array(self.population[best_idx])

    def crossover_with_dup(self, idx1, idx2) -> None:
        # Onepoint crossover with duplicate CITIES in the child
        rand_point = np.random.randint(0, self.chromozome_len)
        self.population[idx1][rand_point:], self.population[idx2][rand_point:] = (
            self.population[idx2][rand_point:],
            self.population[idx1][rand_point:],
        )

    def window_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Generate 2 random indexes then retain those values from parent 2,
        then replace the rest from parent1.
        Ex: parent1 = [1,2,3,4,5]
            parent2 = [4,3,1,2,5]
            random index = 1, 3
            child = [2, 3, 1, 4, 5]
        3 and 1 from parent 2 are crossover
        """
        # generate 2 random indexes
        rand_idx1, rand_idx2 = sorted(np.random.choice(self.chromozome_len, 2))
        child_part = list(parent2[rand_idx1:rand_idx2])
        # fill the child with parent 1
        child = []
        for idx, city in enumerate(parent1):
            if len(child) < rand_idx1:
                if city not in child_part:
                    child.append(city)
            else:
                loop_end_idx = idx
                break
        # append the part of child
        child += child_part
        # # append the end part to complete the new child
        for idx, city in enumerate(parent1[loop_end_idx:]):
            if city not in child_part:
                child.append(city)
        return np.asarray(child)

    def mutation(self, choromozome) -> np.ndarray:
        # generate 2 random indexes
        rand_idx1, rand_idx2 = sorted(np.random.choice(self.chromozome_len, 2))
        choromozome[rand_idx1], choromozome[rand_idx2] = (
            choromozome[rand_idx2],
            choromozome[rand_idx1],
        )
        return choromozome

    def run(self) -> tuple[float, np.ndarray]:
        self.population = self.init_population(
            self.population_size, self.chromozome_len
        )
        self.get_population_fitness()
        # Book-keeping
        track_best_paths = []
        track_best_distances = []

        for gen_idx in range(self.max_generation):
            new_population = []
            # Introduce elitism
            sorted_idx = np.argsort(self.fitness_scores)[-self.elite_size :][::-1]
            for idx in sorted_idx:
                new_population.append(self.population[idx])

            for idx in range(self.elite_size, self.population_size):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                # initialize child with parent 1
                child = parent1.copy()
                # crossover only if less than crossover rate
                if np.random.random() < self.crossover_rate:
                    child = self.window_crossover(parent1, parent2)
                # mutate only if less than mutation rate
                if np.random.random() < self.mutation_rate:
                    child = self.mutation(child)
                new_population.append(child)

            # Store the new population
            self.population = np.array(new_population)
            # get population fitness score
            self.get_population_fitness()
            # Save the data for every generation
            _best_tour = list(self.population[np.argmax(self.fitness_scores)])
            _best_distance = self.get_total_distance(_best_tour)
            track_best_distances.append(_best_distance)
            track_best_paths.append(_best_tour)
            if (gen_idx % (self.max_generation * 0.1)) == 0:
                print(
                    f"{gen_idx} Best distance: {_best_distance}, Best Path: {_best_tour}"
                )

        _best_tour = list(self.population[np.argmax(self.fitness_scores)])
        _best_distance = self.get_total_distance(_best_tour)
        track_best_distances.append(_best_distance)
        track_best_paths.append(_best_tour)
        print(f"Best distance: {_best_distance}, Best Path: {_best_tour}")
        return (track_best_distances, track_best_paths)
