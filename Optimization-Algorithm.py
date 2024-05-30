# optimization_algorithms.py

"""
Optimization Algorithms Module for Energy Consumption Optimization in Smart Grids

This module contains functions for building and running optimization algorithms
to optimize energy consumption and distribution in smart grids.

Techniques Used:
- Linear Programming
- Genetic Algorithms
- Particle Swarm Optimization

Metrics Used:
- Energy efficiency
- Cost reduction
- Load balancing
"""

import numpy as np
from scipy.optimize import linprog
from deap import base, creator, tools, algorithms
from pyswarm import pso
import joblib

class OptimizationAlgorithms:
    def __init__(self):
        """
        Initialize the OptimizationAlgorithms class.
        """
        self.models = {}

    def linear_programming(self, c, A, b, bounds):
        """
        Solve the optimization problem using Linear Programming.
        
        :param c: list, coefficients for the objective function
        :param A: 2D list, coefficients for the inequality constraints
        :param b: list, constants for the inequality constraints
        :param bounds: list of tuples, bounds for the decision variables
        :return: result of the optimization
        """
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        self.models['linear_programming'] = result
        return result

    def genetic_algorithm(self, func, bounds, population_size=50, generations=100, cxpb=0.5, mutpb=0.2):
        """
        Solve the optimization problem using a Genetic Algorithm.
        
        :param func: function, objective function to be minimized
        :param bounds: list of tuples, bounds for the decision variables
        :param population_size: int, size of the population
        :param generations: int, number of generations
        :param cxpb: float, probability of mating two individuals
        :param mutpb: float, probability of mutating an individual
        :return: best individual and its fitness value
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, bounds[:, 0], bounds[:, 1])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(bounds))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", func)

        population = toolbox.population(n=population_size)
        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=False)

        best_ind = tools.selBest(population, 1)[0]
        self.models['genetic_algorithm'] = best_ind
        return best_ind, best_ind.fitness.values

    def particle_swarm_optimization(self, func, lb, ub, swarmsize=100, maxiter=200):
        """
        Solve the optimization problem using Particle Swarm Optimization.
        
        :param func: function, objective function to be minimized
        :param lb: list, lower bounds for the decision variables
        :param ub: list, upper bounds for the decision variables
        :param swarmsize: int, number of particles in the swarm
        :param maxiter: int, maximum number of iterations
        :return: optimal solution and its fitness value
        """
        xopt, fopt = pso(func, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
        self.models['particle_swarm_optimization'] = (xopt, fopt)
        return xopt, fopt

    def save_model(self, model_name, filepath):
        """
        Save the trained model to a file.
        
        :param model_name: str, name of the model to save
        :param filepath: str, path to save the model
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found.")
        joblib.dump(model, filepath)

    def load_model(self, model_name, filepath):
        """
        Load a trained model from a file.
        
        :param model_name: str, name to assign to the loaded model
        :param filepath: str, path to the saved model
        """
        self.models[model_name] = joblib.load(filepath)

if __name__ == "__main__":
    def objective_function(x):
        return x[0]**2 + x[1]**2 + x[2]**2
    
    bounds = [(0, 10), (0, 10), (0, 10)]
    bounds_array = np.array(bounds)

    optimizer = OptimizationAlgorithms()

    # Linear Programming
    c = [1, 2, 3]
    A = [[-1, -1, -1], [1, 1, 1]]
    b = [-10, 10]
    lp_result = optimizer.linear_programming(c, A, b, bounds)
    print("Linear Programming Result:", lp_result)

    # Genetic Algorithm
    ga_result, ga_fitness = optimizer.genetic_algorithm(objective_function, bounds_array)
    print("Genetic Algorithm Result:", ga_result, "Fitness:", ga_fitness)

    # Particle Swarm Optimization
    pso_result, pso_fitness = optimizer.particle_swarm_optimization(objective_function, lb=[0, 0, 0], ub=[10, 10, 10])
    print("Particle Swarm Optimization Result:", pso_result, "Fitness:", pso_fitness)

    # Save models
    optimizer.save_model('linear_programming', 'models/linear_programming_model.pkl')
    optimizer.save_model('genetic_algorithm', 'models/genetic_algorithm_model.pkl')
    optimizer.save_model('particle_swarm_optimization', 'models/particle_swarm_optimization_model.pkl')
