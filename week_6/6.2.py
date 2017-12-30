from random import randint
import random
from functools import reduce
GENERATIONS = 100
INDIVIDUALS = 30
THRESHOLD = 50000

def individu():
    """
        Function for creating an  individu , 
        : return : individu
    """
    individu = [intToBitList(random.randint(0, 64)), intToBitList(random.randint(0, 64)), intToBitList(random.randint(0, 64)), intToBitList(random.randint(0, 64))]
    return individu

def population(amount):
    """
        Function for grading the fitness of an individu , 
        : param amount : amount of individuals in the population to be created
        : return : population
    """
    return [individu() for _ in range(amount)]

def fitness(i):
    """
            Function for grading the fitness of an individu , 
            : param individu : the current individu
            : return : next generation population
    """
    lift = ((getIntFromBits(i[0]) - getIntFromBits(i[1])) ** 2 ) + ((getIntFromBits(i[2]) + getIntFromBits(i[3])) ** 2 ) - ((getIntFromBits(i[0]) - 30) ** 3) - ((getIntFromBits(i[2]) - 40) ** 3)
    return lift

def add(a,b):
    return a+b

def grade(population):
    """
        Function for grading a population , 
        : param population : the current population
        : return : next generation population
    """
    summed = reduce (add,(fitness(x) for x in population), 0 )
    return summed / len(population)


def evolve(population,  retain=0.2, random_select=0.05, mutate=0.01):
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

    graded = [(fitness(x), x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic
    # diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)




    # crossover parents to create offspring
    # print("starting on crossover")
    desired_length = len(population) - len(parents)
    children = []
    while len(children) < desired_length:
        male = randint(0, len(parents) - 1)
        female = randint(0, len(parents) - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male[: half] + female[half:]
            children.append(child)

    # mutate some individuals
    # print("starting on mutation")
    for individual in children:
        if mutate > random.random():
            for gen in individual:
                select = random.randint(0, (len(gen) - 1))
                if gen[select] == 1:
                    gen[select] = 0
                else:
                    gen[select] = 1


    parents.extend(children)
    return parents

def intToBitList(n):
    """
            Function for turning an int in to binairy list representation, 
            : param population : the current population
            : return : bits indexed in a list
        """
    return list(map(int, list('{0:06b}'.format(n))))

def getIntFromBits(n):
    """
            Function for getting an int out of a bit representation, 
            : param population : the current population
            : return : decimal value of bit representation
        """
    n.reverse()
    x  = 1
    acc = 0
    for i in range(len(n)):
        if n[i] == 1:
            acc += x
            x = x * 2

        else:
            x = x * 2

    return acc


p = population (INDIVIDUALS)
fitness_history = [ (grade (p), p )]
for _ in range (GENERATIONS):
    p = evolve (p )
    score = grade (p )
    fitness_history.append((score, p))

for e in fitness_history:
    for individu in e[1]:
        if fitness(individu) > THRESHOLD:
            print(fitness(individu), end=" -> ")
            for field in individu:
                print(getIntFromBits(field), end=" ")
            print()



