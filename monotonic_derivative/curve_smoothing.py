
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import os

def initialize_population(population_size, num_points):
    """
    Initializes the population with random individuals.
    
    Args:
        population_size (int): Number of individuals in the population.
        num_points (int): Number of points in each individual.
    
    Returns:
        numpy.ndarray: Initial population as a 2D array.
    """
    return np.random.randint(0, 100, size=(population_size, num_points))

def calculate_fitness(individual, points, alpha):
    """
    Calculates the fitness of an individual based on the total deviation from the original points
    and the abrupt changes in the smoothed curve.
    
    Args:
        individual (numpy.ndarray): The individual for which the fitness is calculated.
        points (numpy.ndarray): The original points.
        alpha (float): Weighting factor for the total deviation (range: 0 to 1).
    
    Returns:
        float: Fitness value for the individual.
    """
    total_deviation = np.sum(np.abs(individual - points))
    abrupt_changes = np.sum(np.abs(np.diff(np.diff(individual))))
    return - (alpha * total_deviation + (1 - alpha) * abrupt_changes)

def selection(population, points, alpha):
    """
    Selects the top two individuals from the population based on their fitness.
    
    Args:
        population (numpy.ndarray): The current population.
        points (numpy.ndarray): The original points.
        alpha (float): Weighting factor for the total deviation (range: 0 to 1).
    
    Returns:
        numpy.ndarray: The top two individuals from the population.
    """
    fitness = [calculate_fitness(individual, points, alpha) for individual in population]
    indices = np.argsort(fitness)[-2:]
    return population[indices]

def crossover(parent1, parent2):
    """
    Performs crossover between two parents to create two offspring.
    
    Args:
        parent1 (numpy.ndarray): The first parent.
        parent2 (numpy.ndarray): The second parent.
    
    Returns:
        tuple: Two offspring as numpy arrays.
    """
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(individual, mutation_rate):
    """
    Mutates an individual by changing the value of a point with a certain probability.
    
    Args:
        individual (numpy.ndarray): The individual to be mutated.
        mutation_rate (float): The probability of mutating a point (range: 0 to 1).
    
    Returns:
        numpy.ndarray: The mutated individual.
    """
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            individual[i] = np.random.randint(0, 100)
    return individual


def curve_smoothing(points, population_size=100, num_generations=1000, mutation_rate=0.1, alpha=0.5, save_plots=False, output_folder="output"):
    """
    Apply a genetic algorithm to smooth a curve represented by a sequence of points.
    
    Parameters:
    points (numpy.array): An array of Y-coordinates representing the original curve.
    population_size (int, optional): The number of individuals in the population. Default is 100.
    num_generations (int, optional): The number of generations to run the algorithm. Default is 1000.
    mutation_rate (float, optional): The probability of a mutation occurring during reproduction. Default is 0.1.
    alpha (float, optional): The trade-off between smoothness and similarity to the original curve. Default is 0.5.
    save_plots (bool, optional): Whether to save the progress of the algorithm as a GIF. Default is False.
    output_folder (str, optional): The folder to save the progress GIF if `save_plots` is True. Default is "output".
    
    Returns:
    numpy.array: The best individual (smoothed curve) found by the genetic algorithm.
    """
    # Create the output folder if it doesn't exist and save_plots is True
    if save_plots:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    # Initialize the population
    population = initialize_population(population_size, len(points))
    images = []
    
    # Iterate through the generations
    for generation in range(num_generations):
        # Select the parents based on their fitness
        parents = selection(population, points, alpha)
        children = []
        
        # Create the next generation by crossover and mutation
        for i in range(int(population_size / 2)):
            child1, child2 = crossover(parents[0], parents[1])
            children.append(mutation(child1, mutation_rate))
            children.append(mutation(child2, mutation_rate))
        
        # Update the population
        population = np.array(children)
        
        # Save the progress as images for the GIF
        if save_plots and generation % 50 == 0:
            best_individual = selection(population, points, alpha)[0]
            plt.plot(points, 'bo-', label='Original curve')
            plt.plot(best_individual, 'ro-', label='Smoothed curve')
            plt.legend()
            
            if generation == num_generations - 1:
                plt.title('Final curve', fontsize=24, alpha=0.2)
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append(imageio.imread(buf))
            plt.clf()
    
    # Select the best individual from the final population
    best_individual = selection(population, points, alpha)[0]
    
    # Save the progress as a GIF
    if save_plots:
        gif_filename = os.path.join(output_folder, "progress.gif")
        imageio.mimsave(gif_filename, images[:-1] + [images[-1]] * 15, duration=0.2)
    
    return best_individual




