import neat.config
import environment
import neural_controller 
from utils import plot_evolution, draw_net
import random
import numpy as np
import neat
import pickle
import time
import json
import os
import imageio
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

color_map = {
    3: "red",
    4: "blue",
    5: "green",
    6: "yellow",
    7: "purple",
    8: "white",
    9: "cyan",
    10: "black"
}
FREQUENCY_EVAL_RETENTION = 10

# TODO: dont save all the info for deap, just neat handles drifts
# TODO: create two separate classes and use checkpoints
# TODO: base class, DEAP class, NEAT class
# TODO: only NEAT handles drifts
# TODO: make a separte class only for NEAT

class LifelongEvoSwarmExperiment:
    # TODO: we dont load
    def __init__(self,
                name : str = None,
                population_size : int = None,
                env : environment.SwarmForagingEnv = None,
                config_neat : neat.config.Config = None, 
                seed : int = None,
                n_envs : int = 1, # Number of environments for evaluating fitness
                n_workers : int = 1
                ):
        # TODO: on all previous... decide if save only the prev models ore all the prevs
        self.name = name
        self.population_size = population_size
        self.env = env
        self.config_neat = config_neat
        self.seed = seed
        self._current_generation = 0

        self.experiment_name = None
        self.best_individual = None
        self.time_elapsed = None
        self.population = None
        if env is not None:
            self.target_color = env.target_color
        else:
            self.target_color = None
        self.prev_target_colors = [] # TODO: we actually dont need that... use prev_envs.target_color

        available_cores = multiprocessing.cpu_count()
        self.n_workers = n_workers if n_workers <= available_cores else available_cores

        self.n_envs = n_envs
        self.prev_envs = []
        self.prev_models = []

        self.eval_retention = None
        self.reg_type = None
        
        self.logbook_generations = []
        self.logbook_summary = {
            "best": [],
            "best_std": [],
            "best_no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": []
        }

        # Set the seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def drift(self, new_colors, new_target):
        # TODO: validate if we are calling this dirft method properly
        
        self.experiment_name = f"{self.experiment_name}_{color_map[new_target]}" # Add the new target color to the name

        # Save the best individual and the environment from previous drift
        self.prev_models.append(copy.deepcopy(self.best_individual))
        self.prev_envs.append(copy.deepcopy(self.env))
        self.prev_target_colors.append(self.target_color)  
        self.env.change_season(new_colors, new_target)
        self.target_color = new_target

        # Reset stats
        self._current_generation = 0
        self.logbook_generations = []
        self.logbook_summary = {
            "best": [],
            "best_std": [],
            "best_no_penalty": [],
            "id_best": [],
            "avg": [],
            "median": [],
            "std": []
        }

        for prev_target in self.prev_target_colors[-self.n_prev_eval_retention:]:
            self.logbook_summary[f"retention_top_{prev_target}"] = []
            self.logbook_summary[f"id_retention_top_{prev_target}"] = []
            self.logbook_summary[f"retention_top_{prev_target}_std"] = []
            self.logbook_summary[f"retention_pop_{prev_target}"] = []
            self.logbook_summary[f"id_retention_pop_{prev_target}"] = [] 
            self.logbook_summary[f"retention_pop_{prev_target}_std"] = []
    
    def _evaluate_genome(self, genome, config, env, env_seeds, regularization = None):
        # TODO: maybe we dont need these checks... check only once
        if regularization not in [None, "genetic_distance", "weight_protection", "gd", "wp", "functional", "fun"]:
            raise ValueError("Regularization must be one of: genetic_distance, weight_protection, gd, wp, functional.")
        if regularization is not None and self.best_individual is None:
            raise ValueError("Best individual is not set. Run the evolutionary algorithm first.")
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitnesses = []
        
        for seed in env_seeds:
            obs, _ = env.reset(seed=seed)
            fitness = 0.0
            
            while True:
                nn_inputs = env.process_observation(obs)
                nn_outputs = np.array([net.activate(nn_input) for nn_input in nn_inputs])

                actions = (2 * nn_outputs - 1) * env.max_wheel_velocity

                obs, reward, done, truncated, _ = env.step(actions)

                fitness += reward

                if done or truncated:
                    break

            fitnesses.append(fitness)
        
        
        # ----- REGULARIZATION -----
        # Genetic distance penalty
        if regularization == "gd" or regularization == "genetic_distance":
            penalty_distance = 0.0
            for prev_model in self.prev_models[-self.n_prev_models:]:
                config.compatibility_weight_coefficient = 0.6
                config.compatibility_disjoint_coefficient = 1.0
                penalty_distance += prev_model.distance(genome, config) # config.genome_config          

            # penalty = self.reg_lambda * penalty_distance
            penalty = self.reg_lambda * (penalty_distance / len(self.prev_models[-self.n_prev_models:]))
                
        # Weight protection penalty
        if regularization == "wp" or regularization == "weight_protection":
            penalty_wp1 = 0.0
            penalty_wp2 = 0.0
            for prev_model in self.prev_models[-self.n_prev_models:]:
                for c in genome.connections:
                    if c in self.best_individual.connections:
                        penalty_wp1 += (prev_model.connections[c].weight - genome.connections[c].weight) **2
                    else:
                        penalty_wp2 += genome.connections[c].weight ** 2    

            penalty = self.reg_lambda[0] * penalty_wp1 + self.reg_lambda[1] * penalty_wp2
        # ---------------------------
        
        # The fitness is the mean over the various environments
        genome_fitness = np.mean(fitnesses)
        std = np.std(fitnesses)
        
        # Apply the penalty
        if regularization is not None:
            genome_penalized_fitness = genome_fitness - penalty 
        else:
            genome_penalized_fitness = None
        
        return genome_fitness, genome_penalized_fitness, std # TODO !!!!

    def _evaluate_genomes_batch(self, genomes, config, env, env_seeds, regularization = None):
        results = []
        
        for genome_id, genome in genomes:
            fitness, penalized_fitness, std = self._evaluate_genome(genome, config, env, 
                                                               env_seeds, regularization)

            results.append((genome_id, [fitness, penalized_fitness, std]))
        
        return results

    def _evaluate_fitness(self, genomes, config):

        population_stats = {} # key: id, value: {"fitness" : a, "adjusted_fitness": b, "retention_fitness": c, "std": d}

        if self.n_envs == 1:
            env_seeds = [self.seed]
        else:
            env_seeds = [random.randint(0, 1000000) for _ in range(self.n_envs)]

        if self.n_workers > 1:
            # ----- PARALLEL EVALUATION -----
            env_parallel = copy.deepcopy(self.env)
            batch_size = max(1, len(genomes) // self.n_workers)
            genome_batches = [genomes[i:i + batch_size] for i in range(0, len(genomes), batch_size)]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(self._evaluate_genomes_batch, batch, config, 
                                           env_parallel, env_seeds, self.reg_type) 
                        for batch in genome_batches]
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
            
            fitness_map = dict(results)
            for genome_id, genome in genomes:
                fitness, penalized_fitness, std = fitness_map[genome_id]
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = {"fitness": fitness, "penalized_fitness": penalized_fitness, "std": std} 
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = {"fitness":fitness, "std": std} 
            # -------------------------------
        else:
            # ----- SEQUENTIAL EVALUATION -----
            for genome_id, genome in genomes:
                fitness, penalized_fitness, std = self._evaluate_genome(genome, config, self.env, env_seeds,
                                                                   self.reg_type)
                if penalized_fitness is not None:
                    genome.fitness = penalized_fitness # If regularization is applied use penalized fitness
                    population_stats[genome_id] = {"fitness": fitness, "penalized_fitness": penalized_fitness, "std": std}
                else:
                    genome.fitness = fitness
                    population_stats[genome_id] = {"fitness": fitness, "std": std}
            # -------------------------------
        
        best_genome = max(genomes, key=lambda x: x[1].fitness)
        self.logbook_summary["best"].append(best_genome[1].fitness)
        self.logbook_summary["id_best"].append(best_genome[0])
        self.logbook_summary["best_std"].append(population_stats[best_genome[0]]["std"])
        
        if self.reg_type is not None:
            self.logbook_summary["best_no_penalty"].append(population_stats[best_genome[0]]["fitness"])
        
        # ----- EVALUATE RETENTION -----
        if self.eval_retention is not None and self.prev_target_colors is not []:

            # Evaluate genomes on the previous task
            if (self._current_generation % FREQUENCY_EVAL_RETENTION == 0
                or self._current_generation == self._generations - 1): # Evaluate at frequency

                if self.n_envs == 1:
                    env_seeds_r = [self.seed]
                else:
                    env_seeds_r = [random.randint(0, 1000000) for _ in range(self.n_envs)]
                
                for prev_env in self.prev_envs[-self.n_prev_eval_retention:]:
                    prev_target = prev_env.target_color
                    print("prev_env", prev_target)
                    
                    if "population" in self.eval_retention or "pop" in self.eval_retention:
                        eval_genomes = copy.deepcopy(genomes)
                        # Find the best genome on the previous task, reevaluate the population
                        if self.n_workers > 1:
                            # Parallel evaluation
                            # TODO: maybe dont repeat this code
                            env_parallel = copy.deepcopy(prev_env)
                            batch_size = max(1, len(genomes) // self.n_workers)
                            genome_batches = [genomes[i:i + batch_size] for i in range(0, len(genomes), batch_size)]

                            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                                futures = [executor.submit(self._evaluate_genomes_batch, batch, config,
                                                        env_parallel, env_seeds_r) 
                                        for batch in genome_batches]
                                results = []
                                for future in as_completed(futures):
                                    results.extend(future.result())
                            
                            fitness_map = dict(results)
                            for genome_id, genome in eval_genomes:
                                retention_fitness, _, retention_std = fitness_map[genome_id]
                                genome.fitness = retention_fitness
                                population_stats[genome_id][f"retention_{prev_target}"] = retention_fitness # add retention to stats
                                population_stats[genome_id][f"retention_{prev_target}_std"] = retention_std # add retention to stats
                        else:
                            # Sequential evaluation
                            for genome_id, genome in eval_genomes:
                                retention_fitness, _, retention_std = self._evaluate_genome(genome, config, prev_env, env_seeds_r)
                                genome.fitness = retention_fitness
                                population_stats[genome_id][f"retention_{prev_target}"] = retention_fitness # add retention to stats
                                population_stats[genome_id][f"retention_{prev_target}_std"] = retention_std # add retention to stats

                        # eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True) # Take the best genome for retention
                        retention_pop_max = max(eval_genomes, key=lambda x: x[1].fitness)
                        id_retenion_pop = retention_pop_max[0]
                        retention_pop = retention_pop_max[1].fitness
                        self.logbook_summary[f"id_retention_pop_{prev_target}"].append(id_retenion_pop)
                        self.logbook_summary[f"retention_pop_{prev_target}"].append(retention_pop)
                        self.logbook_summary[f"retention_pop_{prev_target}_std"].append(population_stats[id_retenion_pop][f"retention_{prev_target}_std"])
                        print(f"Retention_pop: {retention_pop}")

                    if "top" in self.eval_retention:
                        eval_genomes = copy.deepcopy(genomes)
                        # Take top current genome and evaluate on the previous task
                        # eval_genomes.sort(key=lambda x: x[1].fitness, reverse=True)
                        id_top_genome = best_genome[0]
                        top_genome = best_genome[1]
                        retention_top, _, retention_top_std = self._evaluate_genome(top_genome, config, prev_env, env_seeds_r) 
                        self.logbook_summary[f"id_retention_top_{prev_target}"].append(id_top_genome) 
                        self.logbook_summary[f"retention_top_{prev_target}"].append(retention_top)
                        self.logbook_summary[f"retention_top_{prev_target}_std"].append(retention_top_std)
                        print(f"Retention_top: {retention_top}")      
        # -------------------------------
        
        self.logbook_generations.append(population_stats)

        self._current_generation += 1
        
    
    def _run_neat(self, generations):
        self._generations = generations
        if self.config_neat is None:
            raise ValueError("Neat config object is not set. Set the path to the config file first.")

        stats = neat.StatisticsReporter()
        if self.population is None:
            self.population = neat.Population(self.config_neat)
            self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(stats)

        start = time.time()
        # Run NEAT
        self.best_individual = self.population.run(self._evaluate_fitness, generations)
        end = time.time()
        # fitness_function(list(self.population.items()), self.config) # final evaluation
        self.time_elapsed = end - start
        self.log = stats

    def run_genome(self, id_genome, env, filename = None):
        # TODO: make it prettier
        # TODO: check all this
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.config_neat is None:
            raise ValueError("Neat config object is not set. Set the path to the config file first.")
        
        # Get the id genome from population
        genome = self.population.population[id_genome]
        
        controller = neat.nn.FeedForwardNetwork.create(genome, self.config_neat)

        frames = []
        done = False
        total_reward = 0
        obs, _ = env.reset(seed=None)
        frames.append(env.render(True,False))
        while True:
            inputs = env.process_observation(obs)
            
            outputs = np.array([controller.activate(input) for input in inputs])

            actions = (2 * outputs - 1) * env.max_wheel_velocity # Scale output sigmoid in range of wheel velocity

            obs, reward, done, truncated, info = env.step(actions)
            frames.append(env.render(True,False))
            total_reward += reward

            if done or truncated:
                break
        
        if filename is not None:
            print(f"Reward: {total_reward}")
            imageio.mimsave(f"results/{self.experiment_name}/episode_{filename}.gif", frames, fps = 60)
        
        return total_reward, info
    
    def _save_results(self):
        # Plot stats
        bests = self.log.get_fitness_stat(np.max)
        avgs = self.log.get_fitness_mean()
        medians = self.log.get_fitness_median()
        stds = self.log.get_fitness_stdev()        
        
        # TODO: check this and maybe change
        plot_evolution(bests, avgs = avgs, medians = medians, 
                       filename = f"results/{self.experiment_name}/evolution_plot.png")
        
        # Stats
        self.logbook_summary["avg"] = avgs
        self.logbook_summary["median"] = medians
        self.logbook_summary["std"] = stds

        # Experiment info
        experiment_info = {
            "name": self.experiment_name,
            "time": self.time_elapsed,
            "algorithm": "neat",
            "generations":len(bests),
            "episode_duration": self.env.duration,
            "population_size": self.population_size,
            "target_color": self.target_color,
            "prev_target_colors": self.prev_target_colors,
            "n_agents": self.env.n_agents,
            "n_blocks": self.env.n_blocks,
            "n_colors": self.env.n_colors,
            "colors": self.env.colors,
            "season_colors": self.env.season_colors,
            "repositioning": self.env.repositioning,
            "blocks_in_line": self.env.blocks_in_line,
            "max_wheel_velocity": self.env.max_wheel_velocity,
            "sensor_range": self.env.sensor_range,
            "arena_size": self.env.size,
            "n_workers": self.n_workers,
            "n_env": self.n_envs,
            "n_prev_eval_retention": self.n_prev_eval_retention,
            "n_prev_models" : self.n_prev_models,
            "seed": self.seed,
            "retention_type": self.eval_retention,
            "regularization": self.reg_type,
            "regularization_lambdas": self.reg_lambda,
            "id_best": self.logbook_summary["id_best"][-1],
            "best": bests[-1],
            "best_no_penalty": self.logbook_summary["best_no_penalty"][-1] if self.logbook_summary["best_no_penalty"] else None,
        }
        for prev_target in self.prev_target_colors[-self.n_prev_eval_retention:]:
            experiment_info[f"id_retention_top_{prev_target}"] = self.logbook_summary[f"id_retention_top_{prev_target}"][-1] if self.logbook_summary[f"id_retention_top_{prev_target}"] else None
            experiment_info[f"retention_top_{prev_target}"] = self.logbook_summary[f"retention_top_{prev_target}"][-1] if self.logbook_summary[f"retention_top_{prev_target}"] else None
            experiment_info[f"id_retention_pop_{prev_target}"] = self.logbook_summary[f"id_retention_pop_{prev_target}"][-1] if self.logbook_summary[f"id_retention_pop_{prev_target}"] else None
            experiment_info[f"retention_pop_{prev_target}"] = self.logbook_summary[f"retention_pop_{prev_target}"][-1] if self.logbook_summary[f"retention_pop_{prev_target}"] else None
        
        # --- Test stats ---
        test_stats = {}

        for i in range(self.n_envs):
            
            if i == 0:
                total_reward, info = self.run_genome(experiment_info["id_best"], self.env, filename = "current")
                # Current gif stats
                test_stats["id"] = experiment_info["id_best"]
                test_stats["fitness_ep"] = total_reward
                test_stats["correct_retrieves_ep"] = len(info["correct_retrieves"])
                test_stats["wrong_retrieves_ep"] = len(info["wrong_retrieves"])
                test_stats["info_ep"] = info
                test_stats["fitness"] = total_reward
                test_stats["correct_retrieves"] = len(info["correct_retrieves"])
                test_stats["wrong_retrieves"] = len(info["wrong_retrieves"])
            else:
                total_reward, info = self.run_genome(experiment_info["id_best"], self.env, filename = None)
                # Aggregate current stats
                test_stats["fitness"] += total_reward
                test_stats["correct_retrieves"] += len(info["correct_retrieves"])
                test_stats["wrong_retrieves"] += len(info["wrong_retrieves"])

            # Individual retention
            if self.eval_retention is not None and "top" in self.eval_retention:
                for prev_env in self.prev_envs[-self.n_prev_eval_retention:]:
                    prev_target = prev_env.target_color
                    # this should be the same as best_individaul.. TODO: check
                    if i == 0:
                        total_reward_top, info_top = self.run_genome(experiment_info[f"id_retention_top_{prev_target}"], 
                                                                     prev_env, filename = f"retention_top_{prev_target}")
                         # Top gif stats
                        test_stats[f"id_retention_top_{prev_target}"] = experiment_info[f"id_retention_top_{prev_target}"]
                        test_stats[f"retention_top_{prev_target}_ep"] = total_reward_top
                        test_stats[f"retention_top_correct_retrieves_{prev_target}_ep"] = len(info_top["correct_retrieves"])
                        test_stats[f"retention_top_wrong_retrieves_{prev_target}_ep"] = len(info_top["wrong_retrieves"])
                        test_stats[f"retention_top_info_{prev_target}_ep"] = info_top
                        test_stats[f"retention_top_{prev_target}"] = total_reward_top 
                        test_stats[f"retention_top_correct_retrieves_{prev_target}"] = len(info_top["correct_retrieves"])
                        test_stats[f"retention_top_wrong_retrieves_{prev_target}"] = len(info_top["wrong_retrieves"])
                    else: 
                        total_reward_top, info_top = self.run_genome(experiment_info[f"id_retention_top_{prev_target}"], 
                                                                 prev_env, filename = None)
                        # Aggregate top stats
                        test_stats[f"retention_top_{prev_target}"] += total_reward_top
                        test_stats[f"retention_top_correct_retrieves_{prev_target}"] += len(info_top["correct_retrieves"])
                        test_stats[f"retention_top_wrong_retrieves_{prev_target}"] += len(info_top["wrong_retrieves"])
                
            # Population retention
            if self.eval_retention is not None and ("population" in self.eval_retention or "pop" in self.eval_retention):
                for prev_env in self.prev_envs[-self.n_prev_eval_retention:]:
                    prev_target = prev_env.target_color
                    if experiment_info[f"id_retention_pop_{prev_target}"] not in self.population.population:
                        print(f"Genome {experiment_info[f'id_retention_pop_{prev_target}']} not in population.")
                        continue
                    if i == 0:
                        total_reward_pop, info_pop = self.run_genome(experiment_info[f"id_retention_pop_{prev_target}"], 
                                                                 prev_env, filename = f"retention_pop_{prev_target}")
                        # Pop gif stats
                        test_stats[f"id_retention_pop_{prev_target}"] = experiment_info[f"id_retention_pop_{prev_target}"]
                        test_stats[f"retention_pop_{prev_target}_ep"] = total_reward_pop
                        test_stats[f"retention_pop_correct_retrieves_{prev_target}_ep"] = len(info_pop["correct_retrieves"])
                        test_stats[f"retention_pop_wrong_retrieves_{prev_target}_ep"] = len(info_pop["wrong_retrieves"])
                        test_stats[f"retention_pop_info_{prev_target}_ep"] = info_pop
                        test_stats[f"retention_pop_{prev_target}"] = total_reward_pop
                        test_stats[f"retention_pop_correct_retrieves_{prev_target}"] = len(info_pop["correct_retrieves"])
                        test_stats[f"retention_pop_wrong_retrieves_{prev_target}"] = len(info_pop["wrong_retrieves"])
                    else:
                        total_reward_pop, info_pop = self.run_genome(experiment_info[f"id_retention_pop_{prev_target}"], 
                                                                 prev_env, filename = None)
                        # Aggregate pop stats
                        test_stats[f"retention_pop_{prev_target}"] += total_reward_pop
                        test_stats[f"retention_pop_correct_retrieves_{prev_target}"] += len(info_pop["correct_retrieves"])
                        test_stats[f"retention_pop_wrong_retrieves_{prev_target}"] += len(info_pop["wrong_retrieves"])
        # Average stats               
        test_stats["fitness"] /= self.n_envs
        test_stats["correct_retrieves"] /= self.n_envs
        test_stats["wrong_retrieves"] /= self.n_envs
        for prev_target in self.prev_target_colors[-self.n_prev_eval_retention:]:
            if self.eval_retention is not None and "top" in self.eval_retention:
                test_stats[f"retention_top_{prev_target}"] /= self.n_envs
                test_stats[f"retention_top_correct_retrieves_{prev_target}"] /= self.n_envs
                test_stats[f"retention_top_wrong_retrieves_{prev_target}"] /= self.n_envs
            if self.eval_retention is not None and ("population" in self.eval_retention or "pop" in self.eval_retention):
                # If the key is present
                if f"retention_pop_{prev_target}" in test_stats:
                    test_stats[f"retention_pop_{prev_target}"] /= self.n_envs
                    test_stats[f"retention_pop_correct_retrieves_{prev_target}"] /= self.n_envs
                    test_stats[f"retention_pop_wrong_retrieves_{prev_target}"] /= self.n_envs
        # -------------------------------
        
        # Save the logbooks as json
        with open(f"results/{self.experiment_name}/logbook_summary.json", "w") as f:
            json.dump(self.logbook_summary, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_generations.json", "w") as f:
            json.dump(self.logbook_generations, f, indent=4)
        with open(f"results/{self.experiment_name}/logbook_species.json", "w") as f:
            json.dump(self.log.generation_statistics, f, indent=4)
        with open(f"results/{self.experiment_name}/test.json", "w") as f:
            json.dump(test_stats, f, indent=4)
        # Save experiment info as json
        with open(f"results/{self.experiment_name}/info.json", "w") as f:
            json.dump(experiment_info, f, indent=4)
        # Save the environment as pickle
        with open(f"results/{self.experiment_name}/env.pkl", "wb") as f:
            pickle.dump(self.env, f)
        # Save winner as pickle
        with open(f"results/{self.experiment_name}/best_genome.pkl", "wb") as f:
            pickle.dump(self.best_individual, f)
        # Save the population as pickle
        with open(f"results/{self.experiment_name}/population.pkl", "wb") as f:
            pickle.dump(self.population, f)
        # Save prev models as pickle
        with open(f"results/{self.experiment_name}/prev_models.pkl", "wb") as f:
            pickle.dump(self.prev_models, f)
        # Save prev envs as pickle
        with open(f"results/{self.experiment_name}/prev_envs.pkl", "wb") as f:
            pickle.dump(self.prev_envs, f)
        # Save the neat config file
        with open(f"results/{self.experiment_name}/neat_config.pkl", "wb") as f:
            pickle.dump(self.config_neat, f)  

        # Draw net
        draw_net(self.config_neat, self.best_individual, view=False, filename=f"results/{self.experiment_name}/net", fmt="pdf")
        
    
    # TODO: change name of parameters
    def run(self, generations : int, 
            eval_retention : str = None,
            n_prev_eval_retention : int = 1, 
            regularization_type : str = None, 
            regularization_coefficient = None,
            n_prev_models = 1):

        if self.name is None:
            raise ValueError("Name is not set. Set the name of the experiment first.")
        if self.env is None:
            raise ValueError("Environment is not set. Set the environment first.")
        if self.population_size is None:
            raise ValueError("Population size is not set. Set the population size first.")
        if eval_retention is not None:
            for e in eval_retention:
                if e not in ["top", "population", "pop"]:
                    raise ValueError("Evaluation of retention must be one of: top, population/pop.")
        if eval_retention is not None:
            if self.experiment_name is None or self.prev_target_colors is None:
                raise ValueError("Evaluation of retention not available for static environemnt (before drifts).")
        if regularization_type is not None:
            if regularization_coefficient is None:
                raise ValueError("Regularization coefficient is not set.")
            if regularization_type not in ["gd", "wp", "genetic_distance", "weight_protection", "functional", "fun"]:
                raise ValueError("Regularization of retention must be one of: gd, wp, genetic_distance, weight_protection, functional, fun.")
            if regularization_coefficient in ["gd", "genetic_distance"] and len(regularization_coefficient) != 1:
                raise ValueError("Genetic distance regularization must have one value.")
            if regularization_coefficient in ["wp", "weight_protection"] and len(regularization_coefficient) != 2:
                raise ValueError("Weight protection regularization must have two values.")
            if regularization_coefficient in ["functional", "fun"] and len(regularization_coefficient) != 1:
                raise ValueError("Functional regularization must have one value.")
            if self.experiment_name is None:
                raise ValueError("Regularization for retention not available for static environemnt (before drifts).")
            if self.best_individual is None:
                raise ValueError("Best individual is not set. Run the evolutionary algorithm first (i.e. a static evolution to save a reference model).")
        
        self.eval_retention = eval_retention
        self.n_prev_eval_retention = n_prev_eval_retention
        self.reg_type = regularization_type
        self.reg_lambda = regularization_coefficient
        self.n_prev_models = n_prev_models

        if self.experiment_name is None:
            self.experiment_name = f"{self.name}" \
                f"/neat_{self.env.duration}_{generations}_{self.population_size}_{self.env.n_agents}_{self.env.n_blocks}_{self.n_envs}" \
                f"/seed{self.seed}/{color_map[self.target_color]}" # First evolution (no drift - static)
        
        os.makedirs(f"results/{self.experiment_name}", exist_ok=True) # Create directory for the experiment results
    
        print(f"\n{self.experiment_name}")
        print(f"Running neat with with the following parameters:")
        print(f"Name: {self.name}")
        print(f"Duration of episode: {self.env.duration} in steps (one step is {environment.TIME_STEP} seconds)")
        print(f"Number of generations: {generations}")
        print(f"Population size: {self.population_size}")
        print(f"Number of agents: {self.env.n_agents}")
        print(f"Number of blocks: {self.env.n_blocks}")
        print(f"Colors: {self.env.colors} (target color: {color_map[self.target_color]}, n_colors: {self.env.n_colors})")
        # print(f"Repositioning: {self.env.repositioning}")
        # print(f"Blocks in line: {self.env.blocks_in_line}")
        print(f"Number of environments for evaluation: {self.n_envs}")
        print(f"Number of workers: {self.n_workers}")
        print(f"Seed: {self.seed}")

        self.env.reset(seed=self.seed)
        self.env.render() # Show the environment

        self._run_neat(generations)
        
        self._save_results()
