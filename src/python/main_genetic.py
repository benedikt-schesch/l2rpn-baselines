# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pygad.torchga as torchga
import pygad
from environments.Grid2OpBilevelFlattened import Grid2OpBilevelFlattened
from tqdm import tqdm


# Define the Neural Network
class ActionPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActionPredictor, self).__init__()
        # Define your neural network layers here
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layers(x) * 100


class MyFitnessFunction:
    def __init__(self, env: Grid2OpBilevelFlattened, model):
        self.env = env
        self.model = model

    def evaluate(self, pygad_instace, genome, genome_idx):
        # Set the weights of the neural network to the current genome
        torch.nn.utils.vector_to_parameters(
            torch.tensor(genome), self.model.parameters()
        )

        # Reset the environment
        obs, _ = self.env.reset()
        total_reward = 0

        for t in tqdm(
            range(self.env.get_grid2op_env().chronics_handler.max_timestep())
        ):
            # Use the model to predict the action
            obs = obs.double()
            # print("Dtype", obs.dtype, "\n")
            action = self.model(obs)

            # Step the environment using the predicted action
            obs, reward, done, _, _ = self.env.step(action.detach().numpy())
            total_reward += reward
            if done:
                break

        return env.get_time_step()


# Initialize your environment
env = Grid2OpBilevelFlattened("educ_case14_storage")

# Initialize the neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
model = ActionPredictor(input_size, output_size)
torch_ga = torchga.TorchGA(model=model, num_solutions=10)

# Define the genome length (equal to the number of parameters in the model) and population size
genome_length = sum(p.numel() for p in model.parameters())
population_size = 50

# Initialize the fitness function
fitness_func = MyFitnessFunction(env, model).evaluate
initial_population = torch_ga.population_weights

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 250  # Number of generations.
num_parents_mating = (
    10  # Number of solutions to be selected as parents in the mating pool.
)
initial_population = (
    torch_ga.population_weights
)  # Initial population of network weights


def callback_generation(ga_instance):
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


# Create the genetic algorithm instance
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population=initial_population,
    fitness_func=fitness_func,
    #    parallel_processing=["process", 1],
    on_generation=callback_generation,
)

# Run the genetic algorithm
best_solution, best_fitness = ga_instance.run()

print("Best Fitness: ", best_fitness)

# Set the best weights to the model
best_weights = torch.tensor(best_solution)
torch.nn.utils.vector_to_parameters(best_weights, model.parameters())

# Save the best model
torch.save(model.state_dict(), "best_model.pt")
