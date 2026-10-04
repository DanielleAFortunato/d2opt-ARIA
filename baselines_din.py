import numpy as np
import tsp_problem
import config
import os
import pandas as pd
import time
from lib import aplicar_dinamicidade_transito

# =========================================================================
# FUNÇÕES AUXILIARES DE MOVIMENTAÇÃO
# =========================================================================

def mutacao_ponte_dupla(tour_array):
    """Perturbação 4-opt (Double Bridge) para saltar de mínimos locais."""
    tour = np.copy(tour_array[0])
    n = len(tour)
    if n < 8: return tour_array
    idx = np.sort(np.random.choice(n, 4, replace=False))
    i, j, k, l = idx
    new_tour = np.concatenate([tour[0:i], tour[k:l], tour[j:k], tour[i:j], tour[l:]])
    return new_tour.reshape(1, -1)

def busca_local_2opt(tour_array, current_nfe, max_nfe):
    """2-opt Completo (First Improvement) contabilizando o NFE real."""
    if tsp_problem.dist_matrix is None: return tour_array, current_nfe
    best_tour = np.copy(tour_array[0]).astype(int)
    num_cities = len(best_tour)
    improvement = True
    
    while improvement and current_nfe < max_nfe:
        improvement = False
        current_fit = tsp_problem.tsp_fitness(best_tour.reshape(1, -1))[0]
        current_nfe += 1
        
        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                if current_nfe >= max_nfe:
                    break
                
                new_tour = np.copy(best_tour)
                new_tour[i:j] = best_tour[i:j][::-1]
                
                new_fit = tsp_problem.tsp_fitness(new_tour.reshape(1, -1))[0]
                current_nfe += 1
                
                if new_fit < current_fit:
                    best_tour = new_tour
                    improvement = True
                    break
            if improvement:
                break
                
    return best_tour.reshape(1, -1), current_nfe

# =========================================================================
# CLASSES DOS ALGORITMOS BASELINES DINÂMICOS
# =========================================================================

class DynamicACO:
    def __init__(self, num_ants=None, alpha=None, beta=None, rho=None):
        self.num_ants = num_ants if num_ants is not None else getattr(config, 'ACO_ANTS', 30)
        self.alpha = alpha if alpha is not None else getattr(config, 'ACO_ALPHA', 1.0)
        self.beta = beta if beta is not None else getattr(config, 'ACO_BETA', 4.0)
        self.rho = rho if rho is not None else getattr(config, 'ACO_RHO', 0.1)

    def solve(self, ndim, fitness_func, max_nfe, nfe_intervalo, instancia_path=None, exec_id=0):
        t_inicio = time.time()
        dist_matrix = tsp_problem.dist_matrix
        pheromones = np.ones((ndim, ndim))
        
        best_tour = np.random.permutation(ndim)
        best_fit = fitness_func(best_tour.reshape(1, -1))[0]
        current_nfe = 1
        proxima_mudanca = nfe_intervalo
        
        historico = [[current_nfe, best_fit, time.time() - t_inicio, 0.0, self.num_ants]]
        
        while current_nfe < max_nfe:
            if current_nfe >= proxima_mudanca:
                historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, self.num_ants])
                tsp_problem.dist_matrix = aplicar_dinamicidade_transito(
                    tsp_problem.dist_matrix, config.INTENSIDADE_TRANSITO, config.NUM_BLOQUEIOS
                )
                dist_matrix = tsp_problem.dist_matrix
                best_fit = fitness_func(best_tour.reshape(1, -1))[0]
                current_nfe += 1
                historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, self.num_ants])
                proxima_mudanca += nfe_intervalo
            
            ant_tours = []
            for ant in range(self.num_ants):
                if current_nfe >= max_nfe: break
                unvisited = set(range(ndim))
                current_node = np.random.choice(ndim)
                tour = [current_node]
                unvisited.remove(current_node)
                
                while unvisited:
                    nodes_list = list(unvisited)
                    probs = []
                    for node in nodes_list:
                        tau = pheromones[current_node, node] ** self.alpha
                        eta = (1.0 / max(1e-6, dist_matrix[current_node, node])) ** self.beta
                        probs.append(tau * eta)
                    
                    sum_probs = sum(probs)
                    if sum_probs == 0:
                        next_node = np.random.choice(nodes_list)
                    else:
                        probs = [p / sum_probs for p in probs]
                        next_node = np.random.choice(nodes_list, p=probs)
                    
                    tour.append(next_node)
                    unvisited.remove(next_node)
                    current_node = next_node
                
                ant_tours.append(tour)
            
            if not ant_tours: break
            
            ant_tours = np.array(ant_tours)
            ant_fits = fitness_func(ant_tours)
            current_nfe += len(ant_tours)
            
            min_idx = np.argmin(ant_fits)
            if ant_fits[min_idx] < best_fit:
                best_fit = ant_fits[min_idx]
                best_tour = ant_tours[min_idx].copy()
            
            pheromones *= (1.0 - self.rho)
            for tour, fit in zip(ant_tours, ant_fits):
                for idx in range(ndim - 1):
                    pheromones[tour[idx], tour[idx+1]] += (1.0 / max(1e-6, fit))
                pheromones[tour[-1], tour[0]] += (1.0 / max(1e-6, fit))
                
            historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, self.num_ants])
            
        os.makedirs('execucoes', exist_ok=True)
        instancia_nome = os.path.basename(instancia_path).replace('.tsp', '') if instancia_path else 'default'
        df_log = pd.DataFrame(historico, columns=['NFE', 'Fitness', 'Tempo_Segundos', 'Raio_Medio', 'Tamanho_Populacao'])
        df_log.to_csv(f'execucoes/ACO_exec_{exec_id}_{instancia_nome}.csv', index=False)
        return best_fit

class DynamicRIME:
    def __init__(self, pop_size=50):
        self.pop_size = getattr(config, 'RIME_POP_SIZE', pop_size)

    def solve(self, ndim, fitness_func, max_nfe, nfe_intervalo, instancia_path=None, exec_id=0):
        t_inicio = time.time()
        pop = np.array([np.random.permutation(ndim) for _ in range(self.pop_size)])
        fits = fitness_func(pop)
        current_nfe = len(pop)
        
        idx_best = np.argmin(fits)
        best_global = fits[idx_best]
        best_tour = pop[idx_best].copy()
        
        proxima_mudanca = nfe_intervalo
        historico = [[current_nfe, best_global, time.time() - t_inicio, 0.0, len(pop)]]

        while current_nfe < max_nfe:
            if current_nfe >= proxima_mudanca:
                historico.append([current_nfe, best_global, time.time() - t_inicio, 0.0, len(pop)])
                tsp_problem.dist_matrix = aplicar_dinamicidade_transito(
                    tsp_problem.dist_matrix, config.INTENSIDADE_TRANSITO, config.NUM_BLOQUEIOS
                )
                best_global = fitness_func(best_tour.reshape(1, -1))[0]
                current_nfe += 1
                historico.append([current_nfe, best_global, time.time() - t_inicio, 0.0, len(pop)])
                proxima_mudanca += nfe_intervalo

            E = np.sqrt(max(0.0, 1.0 - (current_nfe / max_nfe)))
            new_pop = []
            for i in range(self.pop_size):
                if np.random.rand() < E:
                    child = self._fusion(pop[i], best_tour)
                else:
                    child = pop[i].copy()
                    a, b = np.random.choice(ndim, 2, replace=False)
                    child[a], child[b] = child[b], child[a]
                new_pop.append(child)
            pop = np.array(new_pop)

            fits = fitness_func(pop)
            current_nfe += len(pop)
            
            idx_best_gen = np.argmin(fits)
            if fits[idx_best_gen] < best_global:
                best_global = fits[idx_best_gen]
                best_tour = pop[idx_best_gen].copy()

            historico.append([current_nfe, best_global, time.time() - t_inicio, 0.0, len(pop)])

        os.makedirs('execucoes', exist_ok=True)
        inst_name = os.path.basename(instancia_path).replace('.tsp', '') if instancia_path else 'default'
        df_log = pd.DataFrame(historico, columns=['NFE', 'Fitness', 'Tempo_Segundos', 'Raio_Medio', 'Tamanho_Populacao'])
        df_log.to_csv(f"execucoes/dados_RIME_{inst_name}_{exec_id}.csv", index=False)
        return best_global

    def _fusion(self, p1, p2):
        size = len(p1)
        cut = size // 2
        child = np.full(size, -1)
        child[:cut] = p2[:cut]
        ptr = cut
        for city in p1:
            if city not in child and ptr < size:
                child[ptr] = city
                ptr += 1
        return child

class DynamicLKH:
    def solve(self, ndim, fitness_func, max_nfe, nfe_intervalo, instancia_path=None, exec_id=0):
        t_inicio = time.time()
        current_tour = np.random.permutation(ndim)
        current_fit = fitness_func(current_tour.reshape(1, -1))[0]
        current_nfe = 1
        best_tour = current_tour.copy()
        best_fit = current_fit
        proxima_mudanca = nfe_intervalo
        
        historico = [[current_nfe, best_fit, time.time() - t_inicio, 0.0, 1]]
        
        while current_nfe < max_nfe:
            if current_nfe >= proxima_mudanca:
                historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, 1])
                tsp_problem.dist_matrix = aplicar_dinamicidade_transito(
                    tsp_problem.dist_matrix, config.INTENSIDADE_TRANSITO, config.NUM_BLOQUEIOS
                )
                current_fit = fitness_func(current_tour.reshape(1, -1))[0]
                best_fit = fitness_func(best_tour.reshape(1, -1))[0]
                current_nfe += 2
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_tour = current_tour.copy()
                historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, 1])
                proxima_mudanca += nfe_intervalo
            
            # Busca local 2-opt na solução corrente
            new_tour_array, current_nfe = busca_local_2opt(current_tour.reshape(1, -1), current_nfe, max_nfe)
            new_fit = fitness_func(new_tour_array)[0]
            current_nfe += 1
            current_tour = new_tour_array.flatten()
            current_fit = new_fit
            
            if current_fit < best_fit:
                best_fit = current_fit
                best_tour = current_tour.copy()
            
            historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, 1])
            
            # Perturbação Double Bridge na solução corrente sem corromper best_tour
            if current_nfe < max_nfe:
                perturbed = mutacao_ponte_dupla(current_tour.reshape(1, -1))
                current_tour = perturbed.flatten()
                current_fit = fitness_func(perturbed)[0]
                current_nfe += 1
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_tour = current_tour.copy()
                historico.append([current_nfe, best_fit, time.time() - t_inicio, 0.0, 1])
                
        os.makedirs('execucoes', exist_ok=True)
        instancia_nome = os.path.basename(instancia_path).replace('.tsp', '') if instancia_path else 'default'
        df_log = pd.DataFrame(historico, columns=['NFE', 'Fitness', 'Tempo_Segundos', 'Raio_Medio', 'Tamanho_Populacao'])
        df_log.to_csv(f'execucoes/LKH_exec_{exec_id}_{instancia_nome}.csv', index=False)
        return best_fit