import networkx as nx
import time
import random
import numpy as np
import math


random.seed(10)


random.seed(10)

def is_independent_set(graph, node_sequence):
    nodeList = list(graph.nodes())
    for i in range(len(node_sequence)):
        for j in range(i + 1, len(node_sequence)):
            if graph.has_edge(nodeList[i], nodeList[j]):
                return False
    return True

# aurrekoaren berdina da, baina independent set-a ez bada independent seta izateko zenbat falta zaion adierazten du
def is_independent_set2(graph, node_sequence):
    connectedElements = 0
    nodeList = list(graph.nodes())
    for i in range(len(node_sequence)):
        for j in range(i + 1, len(node_sequence)):
            if graph.has_edge(nodeList[i], nodeList[j]):
                connectedElements -= 1
    return connectedElements

# independent seta ez bada honakoa era negatiboan kalifikatzen du
def helburuFuntzioa2(graph, node_sequence, tam):
    """
    Evaluate the objective function.
    """
    nodes = [i for i in range(tam) if node_sequence[i] == 1]
    result = is_independent_set2(graph, nodes)
    return len(nodes) if result==0 else result

def helburuFuntzioa(graph, node_sequence, tam):
    """
    Evaluate the objective function.
    """
    nodes = [i for i in range(tam) if node_sequence[i] == 1]
    return len(nodes) if is_independent_set(graph, nodes) else 0

# funtzio eraikitzaile honek soluzio on bat topatzen du. Horretarako nodoak ausazko orden batean hartzen ditu eta ahal izanez gero independent set-ean sartzen ditu
def funtzioEraikitzailea2(problemaTamaina, problema):
    emaitza =  [0  for _ in range(problemaTamaina)]
    indizeak = [i for i in range(problemaTamaina)]
    random.shuffle(indizeak)
    for i in indizeak:
        emaitza[i]=1
        if helburuFuntzioa(problema, emaitza, problemaTamaina) == 0:
            emaitza[i]=0

    return emaitza


# funtzio eraikitzaile honek nodoak independent set-aren barruan sartzen ditu probabilitate bati jarraiuz
def funtzioEraikitzailea(problemaTamaina, problema, probabilitatea = 0.3):
    probabilitatea = 2*np.log(problemaTamaina)/probabilitatea
    return [1 if random.random() < probabilitatea else 0 for _ in range(problemaTamaina)]

#funtzio eraikitzaile honek 0 osatutako bektore bat erabiltzen du bilaketa heuristikoa egiteko.
def funtzioEraikitzailea3(problemaTamaina, problema, probabilitatea = 0.3):
    return [0  for _ in range(problemaTamaina)]

def balioaEman(errenkada, matrize):
    # errenkada batetik agertu ez diren balioen artetik eta probabilitate banaketa jarraituta elementu bat hautatu
    errenk =  [(index, val) for index, val in enumerate(matrize[errenkada])]
    
    # #oraindik txertatu behar ditugun balioekin baino ez gara geratuko
    # emandakobalioak.sort(reverse=True)
    # for i in emandakobalioak:
    #   errenk.pop(i)
    sum=0
    #bektoreko posizio bakoitzean aurrekoenak akumulatuko ditugu
    for i in range(len(errenk)):
      sum += errenk[i][1]
      errenk[i] = (errenk[i][0], sum)

    ausaz = random.uniform(0.0, sum)

    j = 0
    i = errenk[(len(errenk)-1)][0]
    aurkitua = False
    while (not aurkitua) and (j < (len(errenk)-1)):
      if ausaz < errenk[j][1]:
        aurkitua = True
        i = errenk[j][0]
      else:
        j += 1
    return i #hau da txertatuko den elementu berria


def sortFun(enum):
    return enum[1]



def EDAiterazioa(populazioa, N, problema, problemaTam):
    """
    Perform one iteration of the Estimation of Distribution Algorithm (EDA).
    """
    matrize = np.zeros((problemaTam, 2), dtype=float)
    balioak = [helburuFuntzioa2(problema, indiv, problemaTam) for indiv in populazioa]

    # Select top N solutions
    sorted_indices = np.argsort(balioak)[-N:]
    popN = [populazioa[i] for i in sorted_indices]

    # Update frequency matrix
    for permutazio in popN:
        for j in range(problemaTam):
            matrize[j][permutazio[j]] += 1

    # Convert frequency to probability
    matrize = (matrize + 1) / (N + problemaTam)

    # Generate new solutions
    new_population = []
    for _ in range(len(populazioa) - N):
        new_solution = [0] * problemaTam
        for idx in range(problemaTam):
            new_solution[idx] = balioaEman(idx, matrize)
        new_population.append(new_solution)

    return popN + new_population




def EDA(problema, populazioTam, aukeratuTam, edaMotaIter, instantziaBerriKop, hasierakoPopBariantza):
  hasDenbora = time.time()
  hasierakoPop = []
  nodoKop = problema.number_of_nodes()
  for i in range(populazioTam):
    if i < 2:
      berr = funtzioEraikitzailea3(nodoKop, problema)
    else:
      berr =funtzioEraikitzailea(nodoKop, problema)
    hasierakoPop.append(berr)
  ###########################
  instantziaBerriKop -= populazioTam
  instantziaIteraziokoEbal =populazioTam - aukeratuTam
  instantziaBerriSortuakKop = 0
  while instantziaBerriSortuakKop < instantziaBerriKop:
    if instantziaBerriKop - instantziaBerriSortuakKop >= instantziaIteraziokoEbal:
      hasierakoPop = edaMotaIter(hasierakoPop, aukeratuTam, problema, nodoKop)
    else:
      hasierakoPop = edaMotaIter(hasierakoPop, populazioTam - instantziaBerriKop + instantziaBerriSortuakKop, problema, nodoKop)
    
    instantziaBerriSortuakKop += instantziaIteraziokoEbal
    #print(instantziaBerriSortuakKop, len(hasierakoPop))
  #soluzio onena aukeratu:
  emaitza = hasierakoPop[0]
  balioa1 = -1
  balioa2 = 0
  for i in hasierakoPop:
    balioa2 = helburuFuntzioa2(problema, i, nodoKop)
    if balioa2 > balioa1:
      emaitza = i
      balioa1 = balioa2
  #print("bukatu da")
  denbora = time.time()-hasDenbora
  return  helburuFuntzioa2(problema, emaitza, nodoKop), denbora, emaitza





def EDA_deia(args, Grafoa):
  populazioTam, aukeratuTam, instantziaBerriKop, probabilitatea = args
  helburuFuntz, denbora, _ = EDA(Grafoa, populazioTam, aukeratuTam, EDAiterazioa,instantziaBerriKop, probabilitatea)
  return helburuFuntz

def EDA_deia_emaitza_guztiak(args, Grafoa):
  populazioTam, aukeratuTam, instantziaBerriKop, probabilitatea = args
  helburuFuntz, denbora, _ = EDA(Grafoa, populazioTam, aukeratuTam, EDAiterazioa,instantziaBerriKop, probabilitatea)
  return helburuFuntz



# new instances will be created by inserting 0 or 1 into the vector. Those 0 or 1s can be inserted more than once
def generar_vecino(solution):
    zenb = len(solution)
    solution_copy = solution.copy()
    quantity = random.randrange(1, zenb+1)
    positions = random.sample(range(zenb), quantity )
    values = np.random.choice([0,1], size=quantity)
    for pos, val in zip(positions, values):
        solution_copy[pos] = val

    return solution_copy

# new instance will be created by switching the value of an specific position. This position will be selected randomly.
def generar_vecino2(solution):
    zenb = len(solution)
    solution_copy = solution.copy() 
    pos = random.sample(range(zenb), 1 )[0]
    lehengo_balioa = solution_copy[pos]
    solution_copy[pos] = solution_copy[pos] * -1 + 1  # esto es para hacer lo siguiente: si era cero ahora sera 1. Si era 1 ahora será 0.

    return solution_copy, lehengo_balioa, pos

def kalkulatu_degree(pos, grafoa, edge_kop):
    degree = grafoa.degree(pos)

    return degree / edge_kop

def update_temp(temp, cooling_rate=0.95):
    """Update the temperature by multiplying it by the cooling rate."""
    return temp * cooling_rate


def problemaTamaina(problema):
    max_val = (problema.number_of_edges() * problema.number_of_nodes())**2
    min_val = problema.number_of_nodes()
    return max_val, min_val


def calcular_temperatura_inicial(p, problema):
    #p = 0.75  # kalibratu
    goi_borne, behe_borne = problemaTamaina(problema)
    delta_valor = goi_borne - behe_borne
    return -delta_valor / math.log(p)

# este es el tamaó del entorno para el MIS, para el LOP hay que cambiarlo
def tamaño_entorno(n):
    return n

def simulatedAnnealingIte(params, problema, temp_jaitsiera,funtzSortz):
    edge_kop = problema.number_of_edges()
    tamaina = problema.number_of_nodes()
    temp, max_evals, RO, amaiera_temp = params
    n = problema.number_of_nodes()
    graph_nodes = problema.nodes()
    evals = 0
    bigarren = 0
    temp_jaitsiera = temp_jaitsiera
    all_degrees = [kalkulatu_degree(indizea, problema,edge_kop) for indizea in graph_nodes]
    solution = funtzSortz(tamaina, problema)
    localFitness = helburuFuntzioa2(problema, solution, tamaina)
    bestFitness =  localFitness
    best_solution = solution.copy()
    tamaño_entorno_valor = tamaño_entorno(n)
        
    while (max_evals < 0 or evals < max_evals) and temp > amaiera_temp:
        while bigarren < (RO * tamaño_entorno_valor) and  (max_evals < 0 or evals < max_evals):
            evals += 1
            bigarren += 1
            s_p, lehengo_balioa, indizea = generar_vecino2(solution)
            fitness = helburuFuntzioa2(problema, s_p, tamaina)
            AE = fitness - localFitness
            if AE >= 0:
                solution = s_p.copy()
                localFitness = fitness
                if fitness > bestFitness:
                    bestFitness = fitness
                    best_solution = solution.copy()

            else:
                lagindu = random.uniform(0, 1)
                # Calculate the exponent
                if lehengo_balioa == 1:
                    exponent = AE * (1 + all_degrees[indizea]) / temp
                else:
                    exponent = AE * (1 - all_degrees[indizea]) / temp
                # Calculate the probability
                txarrera_prob = math.exp(exponent)
                if txarrera_prob > lagindu:

                    solution = s_p.copy()
        
        bigarren = 0
        temp = update_temp(temp, cooling_rate=temp_jaitsiera)


    return best_solution, bestFitness




def simulatedAnnealing(params, problema, probabilitatea,funtzSortz):
  hasi = time.time()
  solution, bestFitness = simulatedAnnealingIte(params, problema, probabilitatea,funtzSortz)
  end_time = time.time()
  return solution, bestFitness, end_time-hasi


# max evals-ik ez egoteko balio hau -1 ean jarriko dugu
def simulatedAnnealing_deia(args, Grafoa):
  hasTenpProba, iterazioak, ROProba, temp_jaitsiera, amaiera_temp, funtzSortz = args
  #hasTenpProba = 0.75
  #ROProba = 0.2
  #iterazioak = 10000
  _, bestFitness, denbora= simulatedAnnealing([calcular_temperatura_inicial(hasTenpProba, Grafoa), iterazioak, ROProba, amaiera_temp], Grafoa, temp_jaitsiera, funtzSortz)
  return bestFitness






def funtzioEraikitzaile_deia(args, Grafoa):
  hasi = time.time()
  funtzSortz = args[0]
  problemaTamaina = Grafoa.number_of_nodes()
  soluzioa = funtzSortz(problemaTamaina, Grafoa)
  fitness = helburuFuntzioa2(Grafoa, soluzioa, problemaTamaina)
  end_time = time.time()
  denbora = end_time- hasi
  return fitness


