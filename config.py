import numpy as np
import tsp_problem

"""
Configurações da estrutura de dados e colunas
"""
FIT = 0        # Coluna do custo (distância)
FIT_NORM = 1   # Fitness normalizado
RAIO = 2       # Raio de supressão adaptativo
ORIGEM = 3     # ID do operador (AOS)

"""
Configurações Gerais e Experimento
"""
execucoes = 10         # Número de rodadas para média e desvio padrão
MAX_NFE = 30000        # Esforço computacional total (Critério de Parada)
ndim = 150               # Será atualizado automaticamente pelo tsp_problem.py
verbose = 1

funcao = tsp_problem.tsp_fitness  # Define a função de fitness global

# PARA INSERIR DINAMICIDADE
NFE_INTERVALO_DINAMICO = 8000  # O ambiente muda a cada xxx avaliações

"""
Hiperparâmetros da Rede Imune 
"""
celulas = 10            # População inicial (Crescimento dinâmico)
tamanho_maximo = 100    # Teto populacional para ambos
nclones = 3             # Pressão de clonagem moderada 
beta = 5.0              # Fator de decaimento da mutação
d = 0.05                # Taxa de inserção de novas células (Diversidade)
PCT_EXPLOTACAO = 0.80   # 80% melhores usam AOS / 20% piores usam exploração forte

CHANCE_2OPT_MEMETICO = 0.5  

# Controle do AOS (Adaptive Operator Selection)
AOS_P_MIN = 0.1         # Probabilidade mínima para evitar descarte de operadores


"""
Hiperparâmetros Ajustados dos Baselines (Para comparação justa)
"""
# Genetic Algorithm (GA)
GA_POP_SIZE = 50        # Reduzido para permitir mais gerações dentro do NFE
GA_MUTATION_RATE = 0.08 # 8% é um equilíbrio entre busca e estabilidade
GA_ELITE_SIZE = 2       # Preservação das melhores soluções 

# Ant Colony Optimization (ACO)
ACO_ANTS = 30           # Equilíbrio entre exploração e custo por geração
ACO_ALPHA = 1.0         # Peso do feromônio
ACO_BETA = 4.0          # Peso da visibilidade (distância entre cidades)
ACO_RHO = 0.1           # Evaporação lenta (10%) para manter memória de rotas

# Clonalg (AIS Clássico)
CLO_POP_SIZE = 50
CLO_BETA_CLONE = 10     # Aumentado para 10 para maior pressão de refino

"""
Parâmetros Dinâmicos (DTSP)
"""
INTERVALO_DINAMICO = 8000 # (se aplicável)
INTENSIDADE_TRANSITO = 20
NUM_BLOQUEIOS = 25

"""
Parâmetros de Plotagem e Logs
"""
NOME_ARQUIVO_PROBLEMA = "pr76.tsp"  # Mantenha para compatibilidade
INSTANCIAS_PARA_TESTAR = [
    "instance/att48.tsp",    # Mude de 'Instances/' para 'instance/'
    "instance/eil51.tsp",
    "instance/berlin52.tsp",
    "instance/pr76.tsp",
    "instance/kroA100.tsp",
    "instance/ch130.tsp"
]

minimizar = True
valmin = -10000
valmax = 10000
rmin = 0.5
rmax = 20.0
PLOT_ROUTE_INTERVAL = 500