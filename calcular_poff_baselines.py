import numpy as np
import os
import glob
import pandas as pd
import re
import config  # Importa dinamicamente os parâmetros e instâncias do projeto

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXECUCOES_DIR = os.path.join(BASE_DIR, 'execucoes')

def compilar_estatisticas_offline_gap():
    # Extrai metadados do config.py
    instancias_config = getattr(config, 'INSTANCIAS_PARA_TESTAR', ['pr76.tsp'])
    otimo_estatico = getattr(config, 'otimo_conhecido', 108159) # Base para cálculo do Gap caso ausente no CSV
    
    # Nomes limpos das instâncias mapeadas
    nomes_validos = [os.path.basename(i).split('.')[0].lower() for i in instancias_config]
    
    print(f"====================================================")
    print(f" CALCULADOR DE PERFORMANCE OFFLINE METRIC (P_off)")
    print(f" Target Instances: {nomes_validos}")
    print(f"====================================================")
    
    # Mapeia todos os CSVs disponíveis na pasta de execuções
    todos_arquivos = glob.glob(os.path.join(EXECUCOES_DIR, "dados_*.csv"))
    if not todos_arquivos:
        print(f"❌ Erro: Nenhum arquivo de log encontrado em '{EXECUCOES_DIR}'.")
        return

    # Regex para identificar: dados_[ALGORITMO]_[INSTANCIA]_[EXEC_ID].csv
    pattern = re.compile(r"dados_([a-zA-Z0-9\-]+)_([a-zA-Z0-9]+)_(\d+)\.csv")
    
    # Estrutura hierárquica para agrupar dados: [instancia][algoritmo] -> lista de gaps offline
    banco_dados = {}
    
    for arq in todos_arquivos:
        nome_base = os.path.basename(arq)
        match = pattern.search(nome_base)
        
        if match:
            algoritmo = match.group(1)
            instancia = match.group(2).lower()
            
            # Filtra apenas se a instância estiver ativa no config.py
            if instancia in nomes_validos:
                if instancia not in banco_dados:
                    banco_dados[instancia] = {}
                if algoritmo not in banco_dados[instancia]:
                    banco_dados[instancia][algoritmo] = []
                
                try:
                    df = pd.read_csv(arq)
                    if df.empty or 'NFE' not in df.columns:
                        continue
                    
                    # Verificação de colunas: Se o baseline não gravou 'Gap_Percent' direto, 
                    # calcula dinamicamente usando o Best_Fitness e o ótimo conhecido do config
                    if 'Gap_Percent' in df.columns:
                        gaps = df['Gap_Percent'].values
                    elif 'Best_Fitness' in df.columns:
                        # Recupera o ótimo específico se cadastrado no tsp_problem
                        otimo_ref = otimo_estatico if 'pr76' in instancia else config.otimo_conhecido
                        gaps = ((df['Best_Fitness'] - otimo_ref) / otimo_ref) * 100
                    else:
                        continue
                    
                    # CÁLCULO DA INTEGRAL DISCRETA (Média temporal da série de Gaps)
                    poff_run = np.mean(gaps)
                    banco_dados[instancia][algoritmo].append(poff_run)
                    
                except Exception as e:
                    print(f" ⚠️ Falha ao processar arquivo {nome_base}: {e}")

    # --- PROCESSAMENTO E EXIBIÇÃO EM FORMATO CIENTÍFICO ---
    for inst, algos in banco_dados.items():
        print(f"\n#---------------------------------------------------")
        print(f" INSTÂNCIA: {inst.upper()}")
        print(f"#---------------------------------------------------")
        
        # Estrutura para montar a tabela final
        linhas_tabela = []
        
        for algo, lista_gaps in algos.items():
            if not lista_gaps: continue
            
            # Média das rodadas (Offline Average Gap Global do Lote)
            media_lote = np.mean(lista_gaps)
            # Desvio Padrão das rodadas (Variabilidade inter-execuções)
            std_lote = np.std(lista_gaps)
            num_runs = len(lista_gaps)
            
            linhas_tabela.append({
                'Algorithm': algo,
                'Offline Gap (%)': f"{media_lote:.4f}%",
                'Std. Deviation (σ)': f"{std_lote:.4f}%",
                'Runs Audited': num_runs
            })
            
        df_tabela = pd.DataFrame(linhas_tabela)
        if not df_tabela.empty:
            print(df_tabela.to_string(index=False))
            
            # Geração automática do código LaTeX correspondente
            print(f"\n  [Código LaTeX Gerado para a Subseção de Resultados]:")
            print(f"  \\begin{{tabular}}{{lrc}}")
            print(f"  \\hline \\textbf{{Algorithm}} & \\textbf{{Offline Average Gap (\\%)}} & \\textbf{{Std. Deviation ($\\sigma$)}} \\\\ \\hline")
            for row in linhas_tabela:
                print(f"  {row['Algorithm']} & {row['Offline Gap (%)']} & {row['Std. Deviation (σ)']} \\\\")
            print(f"  \\hline")
            print(f"  \\end{{tabular}}")
        else:
            print(" Nenhum dado estatístico robusto pôde ser extraído para esta instância.")

if __name__ == "__main__":
    compilar_estatisticas_offline_gap()