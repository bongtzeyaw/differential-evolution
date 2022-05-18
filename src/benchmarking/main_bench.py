import torch
import time
import algo
import benchmark
from statistics import mean


CR = 0.75
F = 1.5
NP = 50
NG = 100


def main():
    
    dim = 2
    domain = 5  
    
    diff_evol = algo.DiffEvol(CR, F, NP, NG, None, None, None)
    torch.set_printoptions(precision=15)
    
    time_list_iter = []
    time_list_vect = []
    
    accuracy_iter = diff_evol.opti_iter(benchmark.beale, dim, domain, benchmark=True)
    accuracy_vect = diff_evol.opti_vect(benchmark.beale, dim, domain, benchmark=True)
    
    for _ in range (5):
        
        tic_iter = time.perf_counter()
        diff_evol.opti_iter(benchmark.beale, dim, domain)
        tac_iter = time.perf_counter()
        time_value_iter = tac_iter - tic_iter
        time_list_iter.append(time_value_iter)
    
    for _ in range (5):
        tic_vect = time.perf_counter()
        diff_evol.opti_vect(benchmark.beale, dim, domain)
        tac_vect = time.perf_counter()
        time_value_vect = tac_vect - tic_vect
        time_list_vect.append(time_value_vect)

    return "Temps (itéré): " + str(mean(time_list_iter)) + "\n" + "Temps (vectorisé): " + str(mean(time_list_vect)) + "\n" + "Précision (itéré): " + str(accuracy_iter) + "\n" + "Précision (vectorisé): " + str(accuracy_vect)


print(main())

