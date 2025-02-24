import matplotlib.pyplot as plt
import numpy as np
from deap import tools, algorithms
from PIL import Image, ImageDraw
import json
import graphviz

def print_kinematic_matrix():
    # Define the original matrix
    A = np.array([
        [-np.sqrt(3)/2, 0.5, 1],
        [0, -1, 1],
        [np.sqrt(3)/2, -0.5, 1]
    ])
    print("Matrix A:")
    # Calculate the inverted matrix
    A_inv = np.linalg.inv(A)
    # Print the inverted matrix
    print("Inverted Matrix A^-1:")
    print(A_inv)

def create_gif(images, gif_path, duration=0.1, loop=0):
    # Save the images as a gif
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )

def visual_grid_to_image(visual_grid, blocks_info = None):
    # Define the size of each cell in the image
    cell_size = 20
    img_size = (len(visual_grid) * cell_size, len(visual_grid[0]) * cell_size)

    # Create a new image with white background
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)

    # Define colors: red, blue, green, yellow, purple, white, cyan, black
    colors = {
        "\033[91m": (255, 0, 0),    # Red
        "\033[94m": (0, 0, 255),    # Blue
        "\033[92m": (0, 255, 0),    # Green
        "\033[93m": (255, 255, 0),  # Yellow
        "\033[95m": (128, 0, 128),  # Purple
        "\033[0m": (255, 255, 255), # White
        "\033[96m": (0, 255, 255),  # Cyan
        "\033[30m": (0, 0, 0)       # Black
    }

    # Draw the grid
    for y, row in enumerate(visual_grid):
        for x, cell in enumerate(row):
            if cell != ".":
                symbol = cell[:6] # Extract the symbol
                color_code = cell[:5]  # Extract the color code
                
                color = colors.get(color_code, (0, 0, 0))  # Default to black if color not found so agent
                if symbol[-1].isdigit():
                    draw.ellipse(
                        [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                        fill=color, outline=(0, 0, 0)
                    )
                else:
                    draw.rectangle(
                        [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                        fill=color
                    )
    if blocks_info != None:
        # Write on top of the image the number of correct and wrong blocks 
        draw.text((0, 0), f"Correct: {blocks_info[0]} Wrong: {blocks_info[1]}", fill=(0, 0, 0))

    return img

def load_logbook_json(path):
    logbook_path = f"{path}/logbook.json"
    with open(logbook_path, "r") as f:
        logbook = json.load(f)
    return logbook

def load_experiment_json(path):
    experiment_path = f"{path}/experiment.json"
    with open(experiment_path, "r") as f:
        experiment = json.load(f)
    return experiment

def plot_evolution(bests, avgs = None, medians = None, stds = None, completion_fitness = None, filename = None):
    x_values = np.arange(len(np.array(avgs)))
    plt.plot(bests, label="best")
    
    if stds is not None and avgs is not None:
        plt.errorbar(x_values, avgs, yerr=stds, label='avg +- std', alpha=0.6)
    if avgs is not None and stds is None:
        plt.plot(avgs, label='avg', alpha=0.6)
    if medians is not None:
        plt.plot(medians, label='median', color='purple', alpha=0.6)
    if completion_fitness is not None:
        plt.axhline(y=completion_fitness, color='g', linestyle='--', label='completion criterion')
    
    plt.legend(fontsize=12)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
# TODO: maybe put them in a deap python file
def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def eaEvoStick(population, toolbox, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals (elitism)
        elites = toolbox.select(population)

        # Clone the selected individuals
        offspring = (list(map(toolbox.clone, elites)) * (len(population)))[:len(population) - len(elites)]

        # Apply mutation on the offspring
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The new population is composed of the elites and the offspring
        population[:] = elites + offspring

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def selElitistAndTournament(individuals, k, frac_elitist = 0.1, tournsize = 3):
    return tools.selBest(individuals, int(k*frac_elitist)) + tools.selTournament(individuals, int(k*(1-frac_elitist)), tournsize=tournsize)

def inverse_sigmoid(y):
    return np.log(y / (1 - y))

def neat_sigmoid(x):
    return 1 / (1 + np.exp(-4.9 * x))

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    # if graphviz is None:
    #     warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
    #     return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot



