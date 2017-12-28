from random import randint
import random
from functools import reduce
import math
SUM = 36
PRODUCT = 360
GENERATIONS = 100
INDIVIDUALS = 30

def individu():
    """
        Function for creating an  individu , 
        : return : individu
    """
    sumStack = []
    productStack = []
    individu = []
    while len(sumStack) < 5:
        x = randint(1, 10)
        if x not in sumStack:
            sumStack.append(x)

    while len(productStack) < 5:
        x = randint(1, 10)
        if x not in sumStack and x not in productStack:
            productStack.append(x)

    individu.extend(sumStack)
    individu.extend(productStack)

    return individu

def population(amount):
    """
        Function for grading the fitness of an individu , 
        : param amount : amount of individuals in the population to be created
        : return : population
    """
    return [individu() for _ in range(amount)]

def fitness(individu, targetSum, targetProduct):
    """
            Function for grading the fitness of an individu , 
            : param individu : the current individu
            : param targetSum : the value that we are aiming for
            : param targetProduct : the value that we are aiming for
            : return : next generation population
    """
    sum  = reduce(lambda x, y: x+y, individu[:int(len(individu) / 2)] )
    prod = reduce(lambda x, y: x*y, individu[int(len(individu) / 2):] )
    return (abs(targetSum - sum) + abs(targetProduct - prod))

def add(a,b):
    return a+b

def grade(population, targetSum, targetProduct):
    """
        Function for grading a population , 
        : param population : the current population
        : param targetSum : the value that we are aiming for
        : param targetProduct : the value that we are aiming for
        : return : next generation population
    """
    summed = reduce (add,(fitness(x, targetSum, targetProduct) for x in population), 0 )
    return summed / len(population)

def evolve(population, targetSum, targetProduct, retain=0.2, random_select=0.05, mutate=0.01):
    """
    Function for evolving a population , that is , creating
    offspring (next generation population ) from combining
    ( crossover ) the fittest individuals of the current
    population
    
    : param population : the current population
    : param target : the value that we are aiming for
    : param retain : the portion of the population that we
        allow to spawn offspring
    : param random_select : the portion of individuals that
        are selected at random , not based on their score
    : param mutate : the amount of random change we apply to
        new offspring
    : return : next generation population
    """

    graded = [ ( fitness(x, targetSum,targetProduct), x ) for x in population]
    graded = [ x[1] for x in sorted(graded) ]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic
    # diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # crossover parents to create offspring
    #print("starting on crossover")
    desired_length = len(population) - len(parents)
    children = []
    while len(children) < desired_length:
        male = randint(0, len(parents) - 1)
        female = randint(0, len(parents) -1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male[: half] + female[half:]
            children.append(child)

    # mutate some individuals
    #print("starting on mutation")
    for individual in children:
        if mutate > random.random():
            half = int(len(individual) / 2 )
            pos_geneSum = randint(0, (half - 1))
            pos_geneProd = randint(half, (len(individual)  - 1))
            tmp = individual[pos_geneSum]
            individual[pos_geneSum] = individual[pos_geneProd]
            individual[pos_geneProd] = tmp

    parents.extend(children)
    return parents


p = population (INDIVIDUALS)
fitness_history = [ (grade (p, SUM, PRODUCT ), p )]
for _ in range (GENERATIONS):
    p = evolve (p, SUM, PRODUCT )
    score = grade (p, SUM, PRODUCT )
    fitness_history.append((score, p))
    #print(score)
for e in fitness_history:
    print(e)




