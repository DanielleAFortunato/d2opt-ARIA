# ARQUIVO: lib.py
import math
import numpy as np
import config
from config import INTENSIDADE_TRANSITO, NUM_BLOQUEIOS
import tsp_problem

# =========================================================================
# CONFIGURAÇÃO DOS OPERADORES (IDs e Nomes)
# =========================================================================
OP_IDS = {
    "mut_troca": 1,
    "mut_insercao": 2,
    "mut_inversao": 3,
    "mut_2opt_stochastic": 4,
    "perturbacao_ponte_dupla": 5,
    "pai_original": 0,
    "polimento_memetico": 99,
}
OP_NAMES_BY_ID = {v: k for k, v in OP_IDS.items()}

# =========================================================================
# FUNÇÃO DE DINAMICIDADE (SIMULADOR DE TRÂNSITO)
# =========================================================================
def aplicar_dinamicidade_transito(
    matriz_dist,
    intensidade=INTENSIDADE_TRANSITO,
    num_bloqueios=NUM_BLOQUEIOS,
):
  """Simula mudanças dinâmicas no ambiente alterando pesos em arestas aleatórias."""
  n = matriz_dist.shape[0]
  print(
      f"    -> Aplicando {num_bloqueios} bloqueios com intensidade até"
      f" {intensidade}x..."
  )

  bloqueios_aplicados = set()

  for _ in range(num_bloqueios):
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while i == j or tuple(sorted((i, j))) in bloqueios_aplicados:
      j = np.random.randint(0, n)

    bloqueios_aplicados.add(tuple(sorted((i, j))))
    fator = np.random.uniform(2.0, intensidade)

    matriz_dist[i, j] = matriz_dist[i, j] * fator
    matriz_dist[j, i] = matriz_dist[i, j]

  return matriz_dist


# =========================================================================
# DISTÂNCIA DE CAYLEY PARA TSP
# =========================================================================
def _cayley_dist_perm(p1, p2):
  """Calcula a distância de Cayley básica: d = n - c(sigma)."""
  n = len(p1)
  pos = np.empty(n, dtype=int)
  pos[p1] = np.arange(n)
  sigma = pos[p2]

  visited = np.zeros(n, dtype=bool)
  cycles = 0
  for k in range(n):
    if not visited[k]:
      cycles += 1
      curr = k
      while not visited[curr]:
        visited[curr] = True
        curr = sigma[curr]
  return n - cycles


def calcula_distancia_cayley(populacao):
  """Calcula a matriz de distância de Cayley para o TSP.

  Considera a invariância de ponto inicial (cidade 0) e orientação (direta e
  reversa).
  """
  n_individuos = populacao.shape[0]
  matriz_dist = np.zeros((n_individuos, n_individuos))

  # Normalização: todas as rotas iniciam na cidade 0
  pop_norm = np.zeros_like(populacao)
  for i in range(n_individuos):
    idx_zero = np.where(populacao[i] == 0)[0][0]
    pop_norm[i] = np.roll(populacao[i], -idx_zero)

  for i in range(n_individuos):
    p1 = pop_norm[i]
    for j in range(i + 1, n_individuos):
      p2_dir = pop_norm[j]
      p2_rev = np.concatenate(([p2_dir[0]], p2_dir[1:][::-1]))

      d_dir = _cayley_dist_perm(p1, p2_dir)
      d_rev = _cayley_dist_perm(p1, p2_rev)
      dist = min(d_dir, d_rev)

      matriz_dist[i, j] = dist
      matriz_dist[j, i] = dist

  return matriz_dist


def calcula_distancia_discreta(populacao):
  """Interface compatível que redireciona para a métrica de Cayley."""
  return calcula_distancia_cayley(populacao)


# =========================================================================
# FUNÇÕES AUXILIARES DE RAIO, FITNESS E NORMALIZAÇÃO
# =========================================================================
def calcula_raio(populacao, caracteristicas, matriz_distancia=None):
  if matriz_distancia is None:
    distancias = calcula_distancia_discreta(populacao)
  else:
    distancias = np.copy(matriz_distancia)

  distancias[np.where(distancias == 0)] = math.inf
  distancias_vizinhos = np.amin(distancias, 0)

  if np.all(np.isinf(distancias_vizinhos)):
    config.rmax = 1.5
    config.rmin = 0.1
  else:
    dist_finitas = distancias_vizinhos[np.isfinite(distancias_vizinhos)]
    if dist_finitas.size > 0:
      config.rmax = np.amax(dist_finitas) / 2.0
      config.rmin = np.mean(dist_finitas) / 2.0
    else:
      config.rmax = 1.5
      config.rmin = 0.1

  if config.rmin == config.rmax:
    config.rmin = config.rmax * 0.9 if config.rmax > 0 else 0.1

  r = (
      (config.rmax - config.rmin) * caracteristicas[:, config.FIT_NORM]
      + config.rmin
  )
  return r


def _calcula_fitness(funcao, populacao, minimizar, fator):
  if populacao.shape[0] == 0:
    return np.array([]), np.array([]), np.array([]), 0
  vals = funcao(populacao)
  config.numero_avaliacoes += populacao.shape[0]
  desempenhos_comp = -vals if minimizar else vals
  idx_melhor = np.argmax(desempenhos_comp)
  return desempenhos_comp, vals, populacao[idx_melhor], vals[idx_melhor]


def desempenho(funcao, populacao, minimizar, fator):
  if populacao.shape[0] == 0:
    return np.array([]), np.array([]), np.array([]), float("inf")
  return _calcula_fitness(funcao, populacao, minimizar, fator)


def normaliza(caracteristicas):
  vals = caracteristicas[:, config.FIT]
  min_fit = vals.min()
  max_fit = vals.max()
  if max_fit == min_fit:
    return np.zeros_like(vals)
  return (vals - min_fit) / (max_fit - min_fit)


# =========================================================================
# OPERADORES DE MUTAÇÃO E BUSCA LOCAL
# =========================================================================
def mutacao_troca(tour_array):
  clone = np.copy(tour_array)
  n = clone.shape[1]
  i, j = np.random.choice(n, 2, replace=False)
  clone[0, i], clone[0, j] = clone[0, j], clone[0, i]
  return clone


def mutacao_insercao(tour_array):
  clone = np.copy(tour_array[0])
  n = len(clone)
  i = np.random.randint(0, n)
  val = clone[i]
  temp = np.delete(clone, i)
  j = np.random.randint(0, n)
  res = np.insert(temp, j, val)
  return res.reshape(1, -1)


def mutacao_inversao(tour_array):
  clone = np.copy(tour_array[0])
  n = len(clone)
  i, j = np.random.choice(n, 2, replace=False)
  if i > j:
    i, j = j, i
  clone[i : j + 1] = clone[i : j + 1][::-1]
  return clone.reshape(1, -1)


def mutacao_ponte_dupla(tour_array):
  tour = np.copy(tour_array[0])
  n = len(tour)
  idx = np.sort(np.random.choice(n, 4, replace=False))
  i, j, k, l = idx
  if i < j and j < k and k < l:
    return np.concatenate(
        [tour[0:i], tour[k:l], tour[j:k], tour[i:j], tour[l:]]
    ).reshape(1, -1)
  return tour_array


def mutacao_2opt_stochastic(tour_array):
  best_tour = np.copy(tour_array[0])
  current_val = tsp_problem.tsp_fitness(best_tour.reshape(1, -1))[0]
  n = len(best_tour)

  for _ in range(50):
    i, j = np.sort(np.random.choice(n, 2, replace=False))
    new_tour = np.copy(best_tour)
    new_tour[i : j + 1] = new_tour[i : j + 1][::-1]
    new_val = tsp_problem.tsp_fitness(new_tour.reshape(1, -1))[0]

    if new_val < current_val:
      return new_tour.reshape(1, -1)

  return tour_array


def busca_local_2opt(tour_array):
  if tsp_problem.dist_matrix is None:
    return tour_array
  best_tour = np.copy(tour_array[0]).astype(int)
  num_cities = len(best_tour)
  improvement = True

  while improvement:
    improvement = False
    current_fit = tsp_problem.tsp_fitness(best_tour.reshape(1, -1))[0]

    for i in range(num_cities - 1):
      for j in range(i + 2, num_cities):
        if j == num_cities - 1 and i == 0:
          continue

        new_tour = np.copy(best_tour)
        new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
        new_fit = tsp_problem.tsp_fitness(new_tour.reshape(1, -1))[0]

        if new_fit < current_fit:
          best_tour = new_tour
          current_fit = new_fit
          improvement = True
          break
      if improvement:
        break
  return best_tour.reshape(1, -1)


OPERADORES_MAP = {
    "mut_troca": mutacao_troca,
    "mut_insercao": mutacao_insercao,
    "mut_inversao": mutacao_inversao,
    "mut_2opt_stochastic": mutacao_2opt_stochastic,
}
OPERADORES_AOS_NOMES = list(OPERADORES_MAP.keys())


# =========================================================================
# GERA DESCENDENTES (AOS + Estratégia Memética)
# =========================================================================
def gera_descendentes(
    populacao,
    caracteristicas,
    nclones,
    beta,
    valmin,
    valmax,
    funcao,
    minimizar,
    fator,
    selector,
):
  descendentes = np.zeros_like(populacao)
  desc_origins_map = {}

  pct_corte = (1.0 - config.PCT_EXPLOTACAO) * 100
  limiar_dinamico = np.percentile(caracteristicas[:, config.FIT], pct_corte)

  for i, celula in enumerate(populacao):
    celula_2d = celula.reshape(1, -1)
    fit_atual = caracteristicas[i, config.FIT]

    if fit_atual >= limiar_dinamico:
      melhor_clone_local = np.copy(celula_2d)
      melhor_fit_local = fit_atual
      origem = "pai_original"

      clones_candidatos = []
      fits_candidatos = []
      origens_candidatos = []

      for _ in range(int(nclones)):
        op_nome = selector.choose_operator()
        clone = OPERADORES_MAP[op_nome](celula_2d)
        val = funcao(clone)[0]
        fit_val = -val if minimizar else val

        if fit_val > fit_atual:
          selector.register_reward(op_nome, 1)

        clones_candidatos.append(clone)
        fits_candidatos.append(fit_val)
        origens_candidatos.append(op_nome)

      if len(fits_candidatos) > 0:
        idx_best = np.argmax(fits_candidatos)
        best_mutado = clones_candidatos[idx_best]
        fit_mutado = fits_candidatos[idx_best]
        origem_mutado = origens_candidatos[idx_best]

        if np.random.rand() < getattr(config, "CHANCE_2OPT_MEMETICO", 0.5):
          clone_polido = busca_local_2opt(best_mutado)
          val_polido = funcao(clone_polido)[0]
          fit_polido = -val_polido if minimizar else val_polido

          if fit_polido > fit_mutado:
            best_mutado = clone_polido
            fit_mutado = fit_polido
            origem_mutado = "polimento_memetico"

        if fit_mutado > melhor_fit_local:
          melhor_fit_local = fit_mutado
          melhor_clone_local = best_mutado
          origem = origem_mutado

      descendentes[i] = melhor_clone_local[0]
      desc_origins_map[i] = origem

    else:
      clone_p = celula_2d.copy()
      for _ in range(3):
        clone_p = mutacao_ponte_dupla(clone_p)
      descendentes[i] = clone_p[0]
      desc_origins_map[i] = "perturbacao_ponte_dupla"

  pop_total = np.concatenate((populacao, descendentes), axis=0)
  chars_total = np.zeros(
      (pop_total.shape[0], max(config.RAIO, config.ORIGEM) + 1)
  )
  vals = funcao(pop_total)
  chars_total[:, config.FIT] = -vals if minimizar else vals

  n_pais = len(populacao)
  if caracteristicas.shape[1] > config.ORIGEM:
    chars_total[:n_pais, config.ORIGEM] = caracteristicas[:, config.ORIGEM]

  for k in range(n_pais):
    nome_op = desc_origins_map.get(k)
    chars_total[n_pais + k, config.ORIGEM] = OP_IDS.get(nome_op, 0)

  chars_total[:, config.FIT_NORM] = normaliza(chars_total)
  return (pop_total, chars_total, 0.0, desc_origins_map)


# =========================================================================
# GERAÇÃO E SUPRESSÃO DA POPULAÇÃO
# =========================================================================
def gera_populacao(ndim, n_celulas, valmin, valmax, funcao, minimizar, fator):
  base = np.arange(ndim)
  pop = np.zeros((n_celulas, ndim), dtype=int)
  for i in range(n_celulas):
    pop[i] = np.random.permutation(base)
  chars = np.zeros((n_celulas, max(config.RAIO, config.ORIGEM) + 1))
  vals = funcao(pop)
  chars[:, config.FIT] = -vals if minimizar else vals
  chars[:, config.FIT_NORM] = normaliza(chars)
  chars[:, config.RAIO] = calcula_raio(pop, chars)
  chars[:, config.ORIGEM] = 0
  return (pop, chars)


def gera_celulas(ndim, n_base, d, valmin, valmax):
  qtd = int(n_base * d)
  if qtd < 1 and n_base > 0:
    qtd = 1
  if qtd == 0:
    return np.array([]).reshape(0, ndim), np.array([]).reshape(
        0, config.RAIO + 1
    )
  base = np.arange(ndim)
  pop = np.zeros((qtd, ndim), dtype=int)
  for i in range(qtd):
    pop[i] = np.random.permutation(base)
  chars = np.zeros((qtd, max(config.RAIO, config.ORIGEM) + 1))
  return (pop, chars)


def supressao(
    populacao,
    caracteristicas,
    funcao,
    minimizar,
    fator,
    matriz_distancia=None,
):
  if populacao.shape[0] == 0:
    return populacao, caracteristicas
  ordem = np.argsort(-caracteristicas[:, config.FIT])
  pop_ord = populacao[ordem]
  char_ord = caracteristicas[ordem]
  mat_dist = (
      matriz_distancia[ordem][:, ordem]
      if matriz_distancia is not None
      else calcula_distancia_discreta(pop_ord)
  )
  manter = []
  suprimidos = set()
  n = pop_ord.shape[0]
  for i in range(n):
    if i in suprimidos:
      continue
    manter.append(i)
    raio = char_ord[i, config.RAIO]
    for j in range(i + 1, n):
      if j in suprimidos:
        continue
      if mat_dist[i, j] < raio:
        suprimidos.add(j)
  return pop_ord[manter], char_ord[manter]