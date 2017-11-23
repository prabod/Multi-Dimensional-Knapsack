import math, random
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', action='store', dest='file_name',
                    help='Enter the File Name')

results = parser.parse_args()
# constant variables for rates

Population = 100
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 5

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

MaxNodes = 1000000
global pool  # species pool


# parse file
def get_file_input(file_name):
    f = open(file_name, 'r')
    M = int(f.readline())
    N = int(f.readline())
    values = []
    wei = [[] for x in range(N)]
    for i in range(M):
        line = f.readline()
        line = map(int, line.split(' '))
        values.append(line[0])
        for w in range(1, len(line)):
            wei[w - 1].append(line[w])
    bounds = map(int, f.readline().strip().split(' '))

    return M, N, values, wei, bounds


def write_to_file(output):
    with open(results.file_name + '_out_NEAT', 'w') as f:
        for each in output:
            f.write(str(each) + '\n')


# assign parsed values to global variables
M, N, values, w, bounds = get_file_input(results.file_name)
InputSize = len(values)

Inputs = InputSize
Outputs = Inputs


# activation function
def sigmoid(x):
    try:
        return 2 / (1 + math.exp(-4.9 * x)) - 1
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0


# increase the innovation of the pool
def newInnovation():
    pool.innovation += 1
    return pool.innovation


class Pool:
    """
    define the species pool
    """

    def __init__(self):
        self.species = []
        self.generation = 0
        self.innovation = Outputs
        self.currentSpecies = 0
        self.currentGenome = 0
        self.currentFrame = 0
        self.maxFitness = 0
        self.maxOutput = []


class Species:
    """
    define species
    """

    def __init__(self):
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0


class Genome:
    """
    define genome
    """

    def __init__(self):
        self.genes = {}
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = None
        self.maxneuron = 0
        self.globalRank = 0
        self.mutationRates = {"connections": MutateConnectionsChance,
                              "link": LinkMutationChance,
                              "bias": BiasMutationChance,
                              "node": NodeMutationChance,
                              "enable": EnableMutationChance,
                              "disable": DisableMutationChance,
                              "step": StepSize}


class Gene:
    """
    define gene
    """

    def __init__(self):
        self.into = 0
        self.out = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation = 0


class Neuron:
    """
    define neuron
    """

    def __init__(self):
        self.incoming = []
        self.value = 0.0


class Network:
    """
    Generate neural network for a specific genome
    """

    def __init__(self, genome):
        self.neurons = {}
        for i in range(0, Inputs):
            self.neurons[i] = Neuron()

        for j in range(0, Outputs):
            self.neurons[MaxNodes + j + 1] = Neuron()

        for key in genome.genes:
            if genome.genes[key].enabled:
                if genome.genes[key].out in self.neurons:
                    pass
                else:
                    self.neurons[genome.genes[key].out] = Neuron()

                self.neurons[genome.genes[key].out].incoming.append(genome.genes[key])

                if genome.genes[key].into in self.neurons:
                    pass
                else:
                    self.neurons[genome.genes[key].into] = Neuron()

        genome.network = self


# initialise genome
def init_genome():
    genome_init = Genome()
    genome_init.innovation = 1
    genome_init.maxneuron = Inputs
    mutate(genome_init)
    return genome_init


# evaluate the current neural network
def eval_network(network, inputs):
    for i in range(0, Inputs):
        network.neurons[i].value = inputs[i]

    for key, neuron in network.neurons.iteritems():
        total = 0

        for j in range(0, len(neuron.incoming)):
            incoming = neuron.incoming[j]
            other = network.neurons[incoming.into]
            total = total + incoming.weight * other.value

        if len(neuron.incoming) > 0:
            neuron.value = sigmoid(total)

    outputs = []
    for k in range(0, Outputs):
        if network.neurons[MaxNodes + k + 1].value > 0:
            outputs.append(1)
        else:
            outputs.append(0)

    return outputs


# crossover function to breed a new child
def crossover(g1, g2):
    if g2.fitness > g1.fitness:
        g1, g2 = g2, g1
    child = Genome()

    new_innovations = {}
    for i in g2.genes:
        gene = g2.genes[i]
        new_innovations[gene.innovation] = gene

    for i in g1.genes:
        gene1 = g1.genes[i]
        if gene1.innovation in new_innovations:
            gene2 = new_innovations[gene1.innovation]
            if random.randint(0, 1) == 1 and gene2.enabled:
                child.genes[len(child.genes)] = copy_gene(gene2)
        else:
            child.genes[len(child.genes)] = copy_gene(gene1)

    child.maxneuron = max(g1.maxneuron, g2.maxneuron)

    for mutation in g1.mutationRates:
        child.mutationRates[mutation] = g1.mutationRates[mutation]
    return child


def copy_gene(gene):
    g = Gene()
    g.into = gene.into
    g.out = gene.out
    g.weight = gene.weight
    g.enabled = gene.enabled
    g.innovation = gene.innovation
    return g


def copy_genome(genome):
    genome2 = Genome()
    genome2.genes = {key: value for (key, value) in genome.genes.iteritems()}
    genome2.maxneuron = genome.maxneuron
    genome2.mutationRates["connections"] = genome.mutationRates["connections"]
    genome2.mutationRates["link"] = genome.mutationRates["link"]
    genome2.mutationRates["bias"] = genome.mutationRates["bias"]
    genome2.mutationRates["node"] = genome.mutationRates["node"]
    genome2.mutationRates["enable"] = genome.mutationRates["enable"]
    genome2.mutationRates["disable"] = genome.mutationRates["disable"]

    return genome2


# pick a random neuron
def random_neuron(genes, isInput):
    neurons = {}
    if isInput:
        for i in range(0, Inputs):
            neurons[i] = True

    for o in range(0, Outputs):
        neurons[MaxNodes + o + 1] = True
    for i in genes:
        if isInput or genes[i].into > Inputs:
            neurons[genes[i].into] = True
        if isInput or genes[i].out > Inputs:
            neurons[genes[i].out] = True
    count = len(neurons)

    n = random.randint(0, count)

    for k in neurons:
        n -= 1
        if n == 0:
            return k

    return 0


# check whether specific gene contains the link
def contains_link(genes, link):
    for i in genes:
        gene = genes[i]
        if gene.into == link.into and gene.out == link.out:
            return True

    return False


# mutate the weight of the link
def point_mutate(genome):
    step = genome.mutationRates["step"]

    for i in genome.genes:
        gene = genome.genes[i]
        if random.random() < PerturbChance:
            gene.weight += random.random() + step * 2 - step
        else:
            gene.weight = random.random() * 4 - 2


# mutate link
def link_mutate(genome, forceBias):
    neuron1 = random_neuron(genome.genes, True)
    neuron2 = random_neuron(genome.genes, False)

    new_link = Gene()

    if neuron1 <= Inputs and neuron2 <= Inputs:
        return
    if neuron2 <= Inputs:
        neuron2, neuron1 = neuron1, neuron2

    new_link.into = neuron1
    new_link.out = neuron2
    if forceBias:
        new_link.into = Inputs

    if contains_link(genome.genes, new_link):
        return
    new_link.innovation = newInnovation()
    new_link.weight = random.random() * 4 - 2
    genome.genes[len(genome.genes)] = new_link


# mutate node
def node_mutate(genome):
    if len(genome.genes) == 0:
        return

    genome.maxneuron += 1
    gene = genome.genes[random.choice(genome.genes.keys())]
    if not gene.enabled:
        return
    gene.enabled = False

    gene1 = copy_gene(gene)
    gene1.out = genome.maxneuron
    gene1.weight = 1.0
    gene1.innovation = newInnovation()
    gene1.enabled = True
    genome.genes[len(genome.genes)] = gene1

    gene2 = copy_gene(gene)
    gene2.into = genome.maxneuron
    gene2.innovation = newInnovation()
    gene2.enabled = True
    genome.genes[len(genome.genes)] = gene2


# enable or disable mutation
def enable_disable_mutate(genome, enable):
    candidates = {}
    for k in genome.genes:
        if genome.genes[k].enabled != enable:
            candidates[len(candidates)] = genome.genes[k]
    if len(candidates) == 0:
        return

    gene = candidates[random.randint(0, len(candidates) - 1)]
    gene.enabled = not gene.enabled


# mutate genome
def mutate(genome):
    for mutation in genome.mutationRates:
        if random.randint(0, 1) == 1:
            genome.mutationRates[mutation] *= 0.95

        else:
            genome.mutationRates[mutation] *= 1.05263

    if random.random() < genome.mutationRates["connections"]:
        point_mutate(genome)

    p = genome.mutationRates["link"]
    while p > 0:
        if random.random() < p:
            link_mutate(genome, False)
        p -= 1

    p = genome.mutationRates["bias"]
    while p > 0:
        if random.random() < p:
            link_mutate(genome, True)
        p -= 1

    p = genome.mutationRates["node"]
    while p > 0:
        if random.random() < p:
            node_mutate(genome)
        p -= 1

    p = genome.mutationRates["enable"]
    while p > 0:
        if random.random() < p:
            enable_disable_mutate(genome, True)
        p -= 1

    p = genome.mutationRates["disable"]
    while p > 0:
        if random.random() < p:
            enable_disable_mutate(genome, False)
        p -= 1


# return the number of disjoint links
def disjoint(genes1, genes2):
    i1 = {}
    for k in genes1:
        gene = genes1[k]
        i1[gene.innovation] = True

    i2 = {}
    for i in genes2:
        gene = genes2[i]
        i2[gene.innovation] = True

    disjointGenes = 0
    for i in genes1:
        gene = genes1[i]
        if gene.innovation in i2:
            disjointGenes += 1

    for i in genes2:
        gene = genes2[i]
        if not gene.innovation in i1:
            disjointGenes += 1

    n = max(len(genes1), len(genes2))
    if n == 0:
        return 0
    else:
        return disjointGenes / n


def weights(genes1, genes2):
    i2 = {}
    for i in genes2:
        gene = genes2[i]
        i2[gene.innovation] = gene
    total = 0
    coincident = 0
    for i in genes1:
        gene = genes1[i]
        if gene.innovation in i2:
            gene2 = i2[gene.innovation]
            total = total + abs(gene.weight - gene2.weight)
            coincident += 1
    if coincident > 0:
        return total / coincident
    else:
        return 0


def same_species(genome1, genome2):
    dd = DeltaDisjoint * disjoint(genome1.genes, genome2.genes)
    dw = DeltaWeights * weights(genome1.genes, genome2.genes)
    # print dd,dw,DeltaThreshold
    return dd + dw < DeltaThreshold


def rank_globally():
    glob = []
    for s in range(len(pool.species)):
        species = pool.species[s]
        species.genomes.sort(key=lambda x: x.fitness)
        for g in range(len(species.genomes)):
            glob.append(species.genomes[g])
    glob.sort(key=lambda x: x.fitness)
    for g in range(len(glob)):
        glob[g].globalRank = g + 1


def calculate_average_fitness(species):
    total = 0
    for s in range(len(species.genomes)):
        genome = species.genomes[s]
        total += genome.globalRank

    species.averageFitness = total / len(species.genomes)


def total_average_fitness():
    total = 0
    for s in range(len(pool.species)):
        species = pool.species[s]
        total += species.averageFitness

    return total


def cull_species(cut):
    for s in range(len(pool.species)):
        species = pool.species[s]
        species.genomes.sort(key=lambda x: x.fitness, reverse=True)
        remaining = math.ceil(len(species.genomes) / 2.0)
        if cut:
            remaining = 1
        while len(species.genomes) > remaining:
            species.genomes.pop()


def breedChild(species):
    child = {}
    if random.random() < CrossoverChance:
        species.genomes.sort(key=lambda x: x.fitness)
        top30 = int(len(species.genomes) * 40 / 100)
        g1 = species.genomes[random.randint(top30, len(species.genomes) - 1)]
        g2 = species.genomes[random.randint(top30, len(species.genomes) - 1)]
        child = crossover(g1, g2)
    else:
        g = species.genomes[random.randint(0, len(species.genomes) - 1)]
        child = copy_genome(g)

    mutate(child)
    return child


def removeStaleSpecies():
    survived = []
    for s in range(len(pool.species)):
        species = pool.species[s]
        species.genomes.sort(key=lambda x: x.fitness, reverse=True)

        if species.genomes[0].fitness > species.topFitness:
            species.topFitness = species.genomes[0].fitness
            species.staleness = 0
        else:
            species.staleness += 1

        if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness:
            survived.append(species)
    pool.species = survived


def removeWeakSpecies():
    survived = []
    total = total_average_fitness()

    for s in range(len(pool.species)):
        species = pool.species[s]
        breed = math.floor(species.averageFitness / total * Population)
        if breed >= 0:
            survived.append(species)
    pool.species = survived


def addToSpecies(child):
    foundSpecies = False

    for s in range(len(pool.species)):
        species = pool.species[s]
        if not foundSpecies and same_species(child, species.genomes[0]):
            species.genomes.append(child)
            foundSpecies = True

    if not foundSpecies:
        childSpecies = Species()
        childSpecies.genomes.append(child)
        pool.species.append(childSpecies)


def newGeneration():
    cull_species(False)
    rank_globally()
    removeStaleSpecies()
    rank_globally()
    for s in range(len(pool.species)):
        species = pool.species[s]
        calculate_average_fitness(species)
    removeWeakSpecies()
    total = total_average_fitness()
    children = []
    for s in range(len(pool.species)):
        species = pool.species[s]
        breed = math.floor(species.averageFitness / total * Population)
        for i in range(int(breed)):
            children.append(breedChild(species))
    cull_species(True)
    while len(children) + len(pool.species) < Population:
        species = pool.species[random.randint(0, len(pool.species) - 1)]
        children.append(breedChild(species))
    for c in range(len(children)):
        child = children[c]
        addToSpecies(child)
    pool.generation += 1


def initializePool():
    global pool
    pool = Pool()
    for i in range(Population):
        basic = init_genome()
        addToSpecies(basic)
    initializeRun()


def initializeRun():
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    Network(genome)


def evaluateCurrent():
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    inputs = values
    return eval_network(genome.network, inputs)


def nextGenome():
    pool.currentGenome += 1
    if pool.currentGenome >= len(pool.species[pool.currentSpecies].genomes):
        pool.currentGenome = 0
        pool.currentSpecies += 1
        if pool.currentSpecies >= len(pool.species):
            newGeneration()
            pool.currentSpecies = 0


def fitnessAlreadyMeasured():
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    return genome.fitness != 0


def fitnessFunc(output, w, bounds):
    out_vector = np.array(output)
    for i in range(len(w)):
        product = np.multiply(out_vector, w[i])
        if np.sum(product) > bounds[i]:
            return -1

    return np.sum(np.multiply(out_vector, values))


initializePool()
fitnessPool = []
while True:
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    output = evaluateCurrent()
    genome.fitness = fitnessFunc(output, w, bounds)
    if genome.fitness > 0:
        if genome.fitness >= pool.maxFitness:
            pool.maxFitness = genome.fitness
            fitnessPool.append(pool.maxFitness)
            pool.maxOutput = output
        # print "Gen ", pool.generation, " species ", pool.currentSpecies, " genome ", pool.currentGenome, " fitness: ", genome.fitness

        while fitnessAlreadyMeasured():
            nextGenome()
        initializeRun()
    else:
        nextGenome()
        initializeRun()

    measured = 0
    total = 0
    for species in pool.species:
        for genome in species.genomes:
            total += 1
            if genome.fitness != 0:
                measured += 1

    if pool.generation == 1000:
        write_to_file(pool.maxOutput)
        print pool.maxOutput, fitnessPool[-1]
        break
        # print "Gen ", pool.generation, " species ", pool.currentSpecies, " genome ", pool.currentGenome, " fitness: ", genome.fitness, " (", math.floor(
        #     (measured / float(total)) * 100), "%)", " max fitness", pool.maxFitness
