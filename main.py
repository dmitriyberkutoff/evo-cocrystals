from deap import base, creator, tools
from rdkit import Chem
from rdkit.Chem import QED
import random
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def calculate_tabletability(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0,
    qed = QED.qed(mol)
    logp = Chem.Crippen.MolLogP(mol)
    return qed + (1 - abs(logp - 2.5)),

def crossover(ind1, ind2):
    if len(ind1) != len(ind2):
        return ind1, ind2

    point = random.randint(1, len(ind1) - 1)

    ind1_new = creator.Individual(ind1[:point] + ind2[point:])
    ind2_new = creator.Individual(ind2[:point] + ind1[point:])

    if Chem.MolFromSmiles(ind1_new) is None:
        ind1_new = ind1  # Вернуться к оригиналу, если новый не валиден
    if Chem.MolFromSmiles(ind2_new) is None:
        ind2_new = ind2  # Вернуться к оригиналу, если новый не валиден

    return ind1_new, ind2_new

def mutate(ind):
    index = random.randint(0, len(ind) - 1)
    new_char = random.choice(["C", "N", "O", "Cl", "Br", "F", "(", ")", "="])
    new_ind = creator.Individual(ind[:index] + new_char + ind[index + 1:])

    if Chem.MolFromSmiles(new_ind) is None:
        return ind,  # Возврат оригинала, если новый SMILES невалидный
    return new_ind,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", str, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("smiles", random.choice, [
    "CCCO", "CCCC", "CC(=O)", "C=C(C)", "CCN(C)",
])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.smiles)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: calculate_tabletability(ind))
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)

for gen in range(100):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            child1, child2 = toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for i in range(len(offspring)):
        if random.random() < 0.2:
            offspring[i], = toolbox.mutate(offspring[i])

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, filter(lambda ind: Chem.MolFromSmiles(ind) is not None, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

fits = [ind.fitness.values[0] for ind in population]
best_ind = tools.selBest(population, 1)[0]
print(f"Лучший SMILES: {best_ind}, Tabletability Score: {best_ind.fitness.values[0]}")