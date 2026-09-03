import glob
import os
import re
import sys
import time
from adaptive_selector import ProbabilityMatchingSelector
import config
from config import FIT, FIT_NORM, ORIGEM, RAIO
from lib import (
    OP_IDS,
    OP_NAMES_BY_ID,
    OPERADORES_AOS_NOMES,
    aplicar_dinamicidade_transito,
    busca_local_2opt,
    calcula_distancia_cayley,
    calcula_distancia_discreta,
    calcula_raio,
    desempenho,
    gera_celulas,
    gera_descendentes,
    gera_populacao,
    normaliza,
    supressao,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsp_problem
from tsp_problem import carregar_instancia, tsp_fitness

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUCOES_DIR = os.path.join(BASE_DIR, "execucoes")
IMAGENS_DIR = os.path.join(BASE_DIR, "imagens")
os.makedirs(EXECUCOES_DIR, exist_ok=True)
os.makedirs(IMAGENS_DIR, exist_ok=True)


class DualLogger(object):

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = open(filename, "a", encoding="utf-8")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.terminal.flush()


def executar_dopt_aria_completo(instancia_path, exec_id):
  nome_instancia_limpo = os.path.basename(instancia_path).split(".")[0]

  ndim_prob, otimo_prob, coords_prob = carregar_instancia(instancia_path)
  config.ndim = ndim_prob
  config.otimo_conhecido = otimo_prob
  config.funcao = tsp_fitness
  config.fator = np.zeros((1, config.ndim))

  if (
      hasattr(tsp_problem, "matriz_base_original")
      and tsp_problem.matriz_base_original is not None
  ):
    tsp_problem.dist_matrix = np.copy(tsp_problem.matriz_base_original)

  MAX_NFE = getattr(config, "MAX_NFE", 30000)
  NFE_INTERVALO = getattr(config, "NFE_INTERVALO_DINAMICO", 8000)
  proxima_mudanca = NFE_INTERVALO

  config.numero_avaliacoes = 0
  config.geracao_atual = 0

  history_data = []
  selector_aos = ProbabilityMatchingSelector(
      operator_names=OPERADORES_AOS_NOMES
  )

  populacao, caracteristicas = gera_populacao(
      config.ndim,
      config.celulas,
      config.valmin,
      config.valmax,
      config.funcao,
      True,
      config.fator,
  )

  t_inicio_exec = time.time()

  caracteristicas[:, FIT], _, melhor_ind_gen, melhor_fit_gen = desempenho(
      config.funcao, populacao, True, config.fator
  )
  melhor_fit_global = melhor_fit_gen
  melhor_tour_global = melhor_ind_gen.copy()

  gap_inicial = (
      (
          (melhor_fit_global - config.otimo_conhecido)
          / config.otimo_conhecido
          * 100
      )
      if config.otimo_conhecido > 0
      else 0
  )
  history_data.append([
      config.numero_avaliacoes,
      melhor_fit_global,
      gap_inicial,
      len(populacao),
      time.time() - t_inicio_exec,
  ])

  while config.numero_avaliacoes < MAX_NFE:

    # I. GATILHO DINÂMICO
    if config.numero_avaliacoes >= proxima_mudanca:
      gap_pre = (
          (
              (melhor_fit_global - config.otimo_conhecido)
              / config.otimo_conhecido
              * 100
          )
          if config.otimo_conhecido > 0
          else 0
      )
      history_data.append([
          config.numero_avaliacoes,
          melhor_fit_global,
          gap_pre,
          len(populacao),
          time.time() - t_inicio_exec,
      ])

      print(
          f"\n>>> [Execução: {exec_id}][NFE: {config.numero_avaliacoes}] CHOQUE"
          f" DINÂMICO APLICADO ({nome_instancia_limpo}) <<<"
      )

      tsp_problem.dist_matrix = aplicar_dinamicidade_transito(
          tsp_problem.dist_matrix,
          intensidade=getattr(config, "INTENSIDADE_TRANSITO", 20),
          num_bloqueios=getattr(config, "NUM_BLOQUEIOS", 25),
      )

      caracteristicas[:, FIT], _, melhor_ind_gen, melhor_fit_gen = desempenho(
          config.funcao, populacao, True, config.fator
      )

      novo_fit_antigo_melhor = config.funcao(
          melhor_tour_global.reshape(1, -1)
      )[0]
      config.numero_avaliacoes += 1

      if melhor_fit_gen < novo_fit_antigo_melhor:
        melhor_fit_global = melhor_fit_gen
        melhor_tour_global = melhor_ind_gen.copy()
      else:
        melhor_fit_global = novo_fit_antigo_melhor

      gap_pos = (
          (
              (melhor_fit_global - config.otimo_conhecido)
              / config.otimo_conhecido
              * 100
          )
          if config.otimo_conhecido > 0
          else 0
      )
      history_data.append([
          config.numero_avaliacoes,
          melhor_fit_global,
          gap_pos,
          len(populacao),
          time.time() - t_inicio_exec,
      ])

      proxima_mudanca += NFE_INTERVALO

    # II. CICLO EVOLUTIVO DE CLONAGEM E AOS
    populacao, caracteristicas, _, _ = gera_descendentes(
        populacao,
        caracteristicas,
        config.nclones,
        config.beta,
        config.valmin,
        config.valmax,
        config.funcao,
        True,
        config.fator,
        selector=selector_aos,
    )

    # Busca Local Memética no melhor indivíduo
    if np.random.rand() < getattr(config, "CHANCE_2OPT_MEMETICO", 0.5):
      idx_best = np.argmax(caracteristicas[:, FIT])
      tour_refinado = busca_local_2opt(
          populacao[idx_best].reshape(1, -1)
      ).flatten()
      populacao[idx_best] = tour_refinado
      val_refinado = config.funcao(tour_refinado.reshape(1, -1))[0]
      config.numero_avaliacoes += 1
      caracteristicas[idx_best, FIT] = (
          -val_refinado if config.minimizar else val_refinado
      )

    # Supressão com Métrica de Cayley
    mat_dist = calcula_distancia_cayley(populacao)
    caracteristicas[:, RAIO] = calcula_raio(
        populacao, caracteristicas, matriz_distancia=mat_dist
    )
    populacao, caracteristicas = supressao(
        populacao,
        caracteristicas,
        config.funcao,
        True,
        config.fator,
        matriz_distancia=mat_dist,
    )

    # Introdução de Novas Células (Metadinâmica)
    novas_c, novas_char = gera_celulas(
        config.ndim,
        populacao.shape[0],
        config.d,
        config.valmin,
        config.valmax,
    )
    populacao = np.concatenate((populacao, novas_c), axis=0)
    caracteristicas = np.concatenate((caracteristicas, novas_char), axis=0)

    # Atualização de desempenho
    caracteristicas[:, FIT], _, melhor_ind_gen, melhor_fit_gen = desempenho(
        config.funcao, populacao, True, config.fator
    )

    if melhor_fit_gen < melhor_fit_global:
      melhor_fit_global = melhor_fit_gen
      melhor_tour_global = melhor_ind_gen.copy()

    caracteristicas[:, FIT_NORM] = normaliza(caracteristicas)
    selector_aos.update_probabilities()

    atual_gap = (
        (
            (melhor_fit_global - config.otimo_conhecido)
            / config.otimo_conhecido
            * 100
        )
        if config.otimo_conhecido > 0
        else 0
    )
    history_data.append([
        config.numero_avaliacoes,
        melhor_fit_global,
        atual_gap,
        len(populacao),
        time.time() - t_inicio_exec,
    ])

    if populacao.shape[0] > config.tamanho_maximo:
      idx_melhores = np.argsort(-caracteristicas[:, FIT])
      populacao = populacao[idx_melhores[: config.tamanho_maximo]]
      caracteristicas = caracteristicas[idx_melhores[: config.tamanho_maximo]]

    config.geracao_atual += 1

  df_history = pd.DataFrame(
      history_data,
      columns=[
          "NFE",
          "Best_Fitness",
          "Gap_Percent",
          "Pop_Size",
          "Time_Elapsed",
      ],
  )
  csv_nome = os.path.join(
      EXECUCOES_DIR, f"dados_ARIA_{nome_instancia_limpo}_{exec_id}.csv"
  )
  df_history.to_csv(csv_nome, index=False)

  return csv_nome, nome_instancia_limpo


def plotar_rastreamento_gap(
    arquivo_csv, nome_instancia, exec_id, zoom_gap=None
):
  data = pd.read_csv(arquivo_csv)
  fig, ax1 = plt.subplots(figsize=(12, 6))

  ax1.set_xlabel("Computational Effort (NFE)", fontsize=11)
  ax1.set_ylabel("Cost of the Route", color="blue", fontsize=11)
  ax1.plot(
      data["NFE"],
      data["Best_Fitness"],
      color="blue",
      label="Best Fitness",
      linewidth=2,
  )
  ax1.tick_params(axis="y", labelcolor="blue")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Tracking Error / Gap (%)", color="red", fontsize=11)
  ax2.plot(
      data["NFE"],
      data["Gap_Percent"],
      color="red",
      linestyle="--",
      alpha=0.5,
      label="Instant Gap %",
  )
  ax2.tick_params(axis="y", labelcolor="red")

  if len(data) > 0:
    idx_choque = data[
        data["NFE"] >= getattr(config, "NFE_INTERVALO_DINAMICO", 8000)
    ].index
    if len(idx_choque) > 0:
      nfe_c = data["NFE"].iloc[idx_choque[0]]
      gap_c = data["Gap_Percent"].iloc[idx_choque[0]]
      ax2.annotate(
          "Environmental Shock\n(Traffic Insertion)",
          xy=(nfe_c, gap_c),
          xytext=(nfe_c - 5000, gap_c - 50),
          arrowprops=dict(
              facecolor="black", shrink=0.05, width=1, headwidth=6
          ),
          fontsize=9,
          bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
      )

  if zoom_gap is not None:
    ax2.set_ylim(-5, zoom_gap)

  plt.title(
      f"(Tracking Performance) - Instance: {nome_instancia} (Exec: {exec_id})",
      fontsize=12,
      fontweight="bold",
  )
  ax1.grid(True, linestyle=":", alpha=0.6)
  fig.tight_layout()
  plt.savefig(
      os.path.join(IMAGENS_DIR, f"tracking_gap_{nome_instancia}_exec{exec_id}.png"),
      dpi=300,
  )
  plt.close()


def plotar_analise_integrada_gap(
    arquivo_csv, nome_instancia, exec_id, zoom_gap=None
):
  data = pd.read_csv(arquivo_csv)

  nfe = data["NFE"]
  gap = data["Gap_Percent"]
  gap_medio_cumulativo = gap.expanding().mean()

  fig, ax = plt.subplots(figsize=(12, 5))

  ax.plot(
      nfe, gap, color="salmon", alpha=0.4, linestyle=":", label="Instant Gap G(k)"
  )
  ax.plot(
      nfe,
      gap_medio_cumulativo,
      color="darkred",
      linewidth=3,
      label="Average Offline Performance (Integrated Error)",
  )
  ax.fill_between(
      nfe,
      gap_medio_cumulativo,
      color="darkred",
      alpha=0.15,
      label="Cumulative Error Area (Integral Metric)",
  )

  ax.text(
      0.1,
      0.85,
      f"Final Offline Performance: {gap_medio_cumulativo.iloc[-1]:.2f}%",
      transform=ax.transAxes,
      fontsize=10,
      fontweight="bold",
      bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="darkred", alpha=0.8),
  )

  if zoom_gap is not None:
    ax.set_ylim(-5, zoom_gap)

  plt.xlabel("Computational Time (NFE)", fontsize=11)
  plt.ylabel("Residual Error / Relative Gap (%)", fontsize=11)
  plt.title(
      "Spatio-Temporal Analysis of the Offline Error Integral:"
      f" {nome_instancia} (Exec: {exec_id})",
      fontsize=12,
      fontweight="bold",
  )
  plt.grid(True, which="both", linestyle="--", alpha=0.5)
  plt.legend(
      loc="upper right", frameon=True, facecolor="white", framealpha=0.9
  )

  plt.tight_layout()
  caminho_img = os.path.join(
      IMAGENS_DIR, f"analise_integrada_{nome_instancia}_exec{exec_id}.png"
  )
  plt.savefig(caminho_img, dpi=300)
  plt.close()


def plotar_decaimento_pos_choque(arquivo_csv, nome_instancia, exec_id):
  data = pd.read_csv(arquivo_csv)
  intervalo_dinamico = getattr(config, "NFE_INTERVALO_DINAMICO", 8000)

  dados_pos_choque = data[data["NFE"] >= intervalo_dinamico].copy()

  if dados_pos_choque.empty:
    return

  nfe_inicial = dados_pos_choque["NFE"].iloc[0]
  janela_analise = dados_pos_choque[
      dados_pos_choque["NFE"] <= (nfe_inicial + 5000)
  ]

  plt.figure(figsize=(10, 5))
  tempo_relativo = janela_analise["NFE"] - nfe_inicial

  plt.plot(
      tempo_relativo,
      janela_analise["Gap_Percent"],
      color="darkorange",
      linewidth=2.5,
      label="Adaptation Curve",
  )
  plt.fill_between(
      tempo_relativo,
      janela_analise["Gap_Percent"],
      alpha=0.1,
      color="darkorange",
  )

  plt.xlabel(
      "Post-Impact Computational Evaluations ($\Delta$ NFE)", fontsize=11
  )
  plt.ylabel("Residual Error / Gap (%)", fontsize=11)
  plt.title(
      f"Decay Dynamics of Post-Perturbation Error: {nome_instancia} (Exec:"
      f" {exec_id})",
      fontsize=12,
      fontweight="bold",
  )
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend()
  plt.tight_layout()

  caminho_img = os.path.join(
      IMAGENS_DIR, f"decaimento_choque_{nome_instancia}_exec{exec_id}.png"
  )
  plt.savefig(caminho_img, dpi=300)
  plt.close()


def plotar_tempo_de_recuperacao_barras(arquivo_csv, nome_instancia, exec_id):
  data = pd.read_csv(arquivo_csv)
  intervalo_dinamico = getattr(config, "NFE_INTERVALO_DINAMICO", 8000)
  max_nfe = getattr(config, "MAX_NFE", 30000)

  pontos_choque = sorted(
      data[data["NFE"] % intervalo_dinamico == 0]["NFE"].unique()
  )
  pontos_choque = [p for p in pontos_choque if p > 0 and p < max_nfe - 1000]

  if not pontos_choque:
    diffs = np.diff(data["Gap_Percent"])
    indices_picos = np.where(diffs > 50)[0]
    pontos_choque = data["NFE"].iloc[indices_picos].values.tolist()

  tempos_recuperacao = []
  nomes_choques = []

  for idx, p_choque in enumerate(pontos_choque):
    dados_janela = data[
        (data["NFE"] >= p_choque) & (data["NFE"] < p_choque + intervalo_dinamico)
    ]
    if len(dados_janela) < 5:
      continue

    nfe_estabilizou = dados_janela["NFE"].iloc[-1]
    for i in range(1, len(dados_janela) - 1):
      custo_atual = dados_janela["Best_Fitness"].iloc[i]
      custo_futuro = dados_janela["Best_Fitness"].iloc[i + 1]
      if abs(custo_atual - custo_futuro) / custo_atual < 0.0005:
        nfe_estabilizou = dados_janela["NFE"].iloc[i]
        break

    nfe_gasto = nfe_estabilizou - p_choque
    tempos_recuperacao.append(nfe_gasto)
    nomes_choques.append(f"Shock {idx+1}")

  if tempos_recuperacao:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        nomes_choques,
        tempos_recuperacao,
        color="teal",
        alpha=0.8,
        edgecolor="darkslategray",
        width=0.5,
    )

    for bar in bars:
      yval = bar.get_height()
      plt.text(
          bar.get_x() + bar.get_width() / 2.0,
          yval + (max(tempos_recuperacao) * 0.02),
          f"{int(yval)} NFE",
          ha="center",
          va="bottom",
          fontweight="bold",
      )

    plt.ylabel("Computational Effort Expenditure (NFE)", fontsize=11)
    plt.title(
        "System Response Time for Critical Events:"
        f" {nome_instancia} (Exec: {exec_id})",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    caminho_img = os.path.join(
        IMAGENS_DIR, f"tempo_resposta_{nome_instancia}_exec{exec_id}.png"
    )
    plt.savefig(caminho_img, dpi=300)
    plt.close()


def plotar_individual_e_media_auto(nome_instancia_limpo, intervalo_dinamico):
  padrao = os.path.join(EXECUCOES_DIR, f"dados_ARIA_{nome_instancia_limpo}_*.csv")
  arquivos = sorted(glob.glob(padrao))

  if not arquivos:
    return

  max_nfes = []
  lista_dfs = []
  for arq in arquivos:
    df = pd.read_csv(arq)
    if not df.empty and "NFE" in df.columns:
      max_nfes.append(df["NFE"].max())
      lista_dfs.append(df)

  if not lista_dfs:
    return

  limite_nfe = int(min(max_nfes))
  nfe_grid = np.linspace(0, limite_nfe, 1200)

  plt.figure(figsize=(12, 6.5))
  gaps_interp_matrix = []

  for idx, df in enumerate(lista_dfs):
    gap_interp = np.interp(nfe_grid, df["NFE"], df["Gap_Percent"])
    gaps_interp_matrix.append(gap_interp)
    plt.plot(
        nfe_grid,
        gap_interp,
        color="dodgerblue",
        linewidth=1.0,
        alpha=0.22,
        label="Individual Run" if idx == 0 else "",
    )

  gaps_matrix = np.array(gaps_interp_matrix)
  mean_gap = np.mean(gaps_matrix, axis=0)
  plt.plot(
      nfe_grid,
      mean_gap,
      color="navy",
      linewidth=2.5,
      label="Batch Mean (Tracking Trend)",
  )

  pontos_choque = [
      p for p in range(intervalo_dinamico, limite_nfe, intervalo_dinamico)
  ]
  for choque in pontos_choque:
    plt.axvline(x=choque, color="crimson", linestyle=":", alpha=0.6, linewidth=1.5)

  plt.xlabel("Computational Effort (NFE)", fontsize=11)
  plt.ylabel("Tracking Error / Relative Gap (%)", fontsize=11)
  plt.title(
      "Multi-run Profile & Central Trend Analysis:"
      f" {nome_instancia_limpo.upper()}",
      fontsize=12,
      fontweight="bold",
  )
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.legend(
      loc="upper right", frameon=True, facecolor="white", framealpha=0.9
  )

  plt.tight_layout()
  caminho_img = os.path.join(
      IMAGENS_DIR,
      f"convergencia_individual_media_{nome_instancia_limpo}.png",
  )
  plt.savefig(caminho_img, dpi=300)
  plt.close()
  print(
      "--> [CONSOLIDADO] Gráfico de Média + Individuais salvo com sucesso em:"
      f" {caminho_img}"
  )


if __name__ == "__main__":
  log_file = os.path.join(EXECUCOES_DIR, "log_multi_instancia.txt")
  sys.stdout = DualLogger(log_file)

  instancias = getattr(
      config,
      "INSTANCIAS_PARA_TESTAR",
      [getattr(config, "NOME_ARQUIVO_PROBLEMA", "instance/pr76.tsp")],
  )
  num_rodadas = getattr(config, "execucoes", 1)
  intervalo_dinamico = getattr(config, "NFE_INTERVALO_DINAMICO", 8000)

  for instancia in instancias:
    nome_limpo = os.path.basename(instancia).split(".")[0]
    print(f"\n=====================================================================")
    print(f">>> INICIANDO EXPERIMENTO EVOLUTIVO: {instancia} <<<")
    print(f"=====================================================================")

    gaps_offline_acumulados = []

    for r in range(num_rodadas):
      print(f"\n--- Rodada {r+1} de {num_rodadas} ---")
      try:
        csv_res, _ = executar_dopt_aria_completo(instancia, r)

        plotar_rastreamento_gap(csv_res, nome_limpo, r)
        plotar_analise_integrada_gap(csv_res, nome_limpo, r)
        plotar_decaimento_pos_choque(csv_res, nome_limpo, r)
        plotar_tempo_de_recuperacao_barras(csv_res, nome_limpo, r)

        df_final = pd.read_csv(csv_res)
        gap_rodada = df_final["Gap_Percent"].mean()
        gaps_offline_acumulados.append(gap_rodada)

      except Exception as e:
        print(f"ERRO na Rodada {r+1} da instância {instancia}: {e}")

    if gaps_offline_acumulados:
      media_gap = np.mean(gaps_offline_acumulados)
      std_gap = np.std(gaps_offline_acumulados)
      print(
          f"\n>>> RESULTADO ESTATÍSTICO COMPILADO PARA {nome_limpo.upper()} <<<"
      )
      print(f" - Média do Offline Average Gap: {media_gap:.4f}%")
      print(f" - Desvio Padrão do Offline Gap: {std_gap:.4f}%")

      plotar_individual_e_media_auto(nome_limpo, intervalo_dinamico)