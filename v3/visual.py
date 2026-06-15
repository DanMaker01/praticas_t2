import os
import json
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from itertools import product

# Criar pasta para salvar as imagens
PASTA_SOLUCOES = "solucoes_validas"
os.makedirs(PASTA_SOLUCOES, exist_ok=True)
os.makedirs(f"{PASTA_SOLUCOES}/imagens", exist_ok=True)
os.makedirs(f"{PASTA_SOLUCOES}/dados", exist_ok=True)

# Dimensões do contêiner
L, W, H = 5, 5, 3

# Dados das caixas: (id, l, w, h, d, nome)
caixas = [
    (1, 1, 1, 1, 0, "Caixa1_1x1x1"),      # Prioridade 0 (sai primeiro)
    (2, 4, 1, 1, 1, "Caixa2_4x1x1"),      # Prioridade 1 (sai segundo)
    (3, 4, 1, 1, 2, "Caixa3_4x1x1"),      # Prioridade 2 (sai terceiro)
]

# Parâmetros da função objetivo
ALPHA = 1.0      # peso para utilização de volume
BETA = 100.0     # peso para penalização de instabilidade (severa)
GAMMA = 10.0     # peso para penalização de ordem de entrega

def gerar_posicoes(l, w, h, L, W, H):
    """Gera todas posições (x,y,z,r) possíveis para uma caixa"""
    posicoes = []
    for r in [0, 1]:
        l_ef = l if r == 0 else w
        w_ef = w if r == 0 else l
        h_ef = h
        
        if l_ef > L or w_ef > W or h_ef > H:
            continue
            
        for x in range(L - l_ef + 1):
            for y in range(W - w_ef + 1):
                for z in range(H - h_ef + 1):
                    posicoes.append((x, y, z, r))
    return list(set(posicoes))

def sobrepoe(pos1, l1, w1, h1, pos2, l2, w2, h2):
    """Verifica se duas caixas se sobrepõem"""
    x1, y1, z1, r1 = pos1
    x2, y2, z2, r2 = pos2
    
    l1_ef = l1 if r1 == 0 else w1
    w1_ef = w1 if r1 == 0 else l1
    h1_ef = h1
    
    l2_ef = l2 if r2 == 0 else w2
    w2_ef = w2 if r2 == 0 else l2
    h2_ef = h2
    
    x_over = not (x1 + l1_ef <= x2 or x2 + l2_ef <= x1)
    y_over = not (y1 + w1_ef <= y2 or y2 + w2_ef <= y1)
    z_over = not (z1 + h1_ef <= z2 or z2 + h2_ef <= z1)
    
    return x_over and y_over and z_over

def verifica_todas_sobreposicoes(posicoes, caixas_info):
    """Verifica se há qualquer sobreposição entre as caixas"""
    n = len(posicoes)
    for i in range(n):
        for j in range(i+1, n):
            if sobrepoe(posicoes[i], caixas_info[i][1], caixas_info[i][2], caixas_info[i][3],
                        posicoes[j], caixas_info[j][1], caixas_info[j][2], caixas_info[j][3]):
                return True
    return False

def verifica_estabilidade_geral(posicoes, caixas_info):
    """Verifica se todas as caixas estão completamente apoiadas"""
    n = len(posicoes)
    for i in range(n):
        x, y, z, r = posicoes[i]
        
        if z == 0:
            continue
        
        l, w, h = caixas_info[i][1], caixas_info[i][2], caixas_info[i][3]
        l_ef = l if r == 0 else w
        w_ef = w if r == 0 else l
        
        # Verificar cada ponto da base da caixa
        for dx in range(l_ef):
            for dy in range(w_ef):
                ponto_suportado = False
                for j in range(n):
                    if i == j:
                        continue
                    xo, yo, zo, ro = posicoes[j]
                    lo, wo, ho = caixas_info[j][1], caixas_info[j][2], caixas_info[j][3]
                    lo_ef = lo if ro == 0 else wo
                    wo_ef = wo if ro == 0 else lo
                    ho_ef = ho
                    
                    # Verifica se o ponto (x+dx, y+dy) está apoiado na caixa j
                    if (xo <= x + dx < xo + lo_ef and 
                        yo <= y + dy < yo + wo_ef and 
                        zo + ho_ef == z):
                        ponto_suportado = True
                        break
                
                if not ponto_suportado:
                    return False  # ponto sem suporte = instável
    return True

def verifica_ordem_entrega(posicoes, caixas_info):
    """Verifica se a ordem de entrega é respeitada (prioridades menores mais à esquerda)"""
    n = len(posicoes)
    for i in range(n):
        for j in range(i+1, n):
            d_i = caixas_info[i][4]
            d_j = caixas_info[j][4]
            x_i = posicoes[i][0]
            x_j = posicoes[j][0]
            
            if d_i < d_j and x_i > x_j:
                return False
            if d_j < d_i and x_j > x_i:
                return False
    return True

def calcula_volume_total_ocupado(posicoes, caixas_info):
    """Calcula o volume total ocupado pelas caixas"""
    volume = 0
    for i, pos in enumerate(posicoes):
        _, l, w, h, _, _ = caixas_info[i]
        x, y, z, r = pos
        l_ef = l if r == 0 else w
        w_ef = w if r == 0 else l
        volume += l_ef * w_ef * h
    return volume

def calcula_penalidade_instabilidade(posicoes, caixas_info):
    """Calcula penalidade para caixas instáveis (não completamente apoiadas)"""
    penalidade = 0
    n = len(posicoes)
    
    for i in range(n):
        x, y, z, r = posicoes[i]
        
        if z == 0:
            continue  # caixa no chão está estável
        
        l, w, h = caixas_info[i][1], caixas_info[i][2], caixas_info[i][3]
        l_ef = l if r == 0 else w
        w_ef = w if r == 0 else l
        
        # Área total da base da caixa
        area_total = l_ef * w_ef
        area_apoiada = 0
        
        # Calcular área apoiada
        for dx in range(l_ef):
            for dy in range(w_ef):
                for j in range(n):
                    if i == j:
                        continue
                    xo, yo, zo, ro = posicoes[j]
                    lo, wo, ho = caixas_info[j][1], caixas_info[j][2], caixas_info[j][3]
                    lo_ef = lo if ro == 0 else wo
                    wo_ef = wo if ro == 0 else lo
                    ho_ef = ho
                    
                    if (xo <= x + dx < xo + lo_ef and 
                        yo <= y + dy < yo + wo_ef and 
                        zo + ho_ef == z):
                        area_apoiada += 1
                        break
        
        # Penalidade proporcional à área não apoiada
        area_nao_apoiada = area_total - area_apoiada
        if area_nao_apoiada > 0:
            # Penalidade severa: quanto maior a área não apoiada, maior a penalidade
            penalidade += (area_nao_apoiada / area_total) * 100  # penalidade máxima 100 por caixa
    
    return penalidade

def calcula_penalidade_ordem_entrega(posicoes, caixas_info):
    """
    Penaliza caixas com prioridade maior (sair depois) que estão embaixo 
    ou atrás de caixas com prioridade menor (sair antes)
    """
    penalidade = 0
    n = len(posicoes)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            d_i = caixas_info[i][4]  # prioridade da caixa i
            d_j = caixas_info[j][4]  # prioridade da caixa j
            
            x_i, y_i, z_i, r_i = posicoes[i]
            x_j, y_j, z_j, r_j = posicoes[j]
            
            l_i, w_i, h_i = caixas_info[i][1], caixas_info[i][2], caixas_info[i][3]
            l_j, w_j, h_j = caixas_info[j][1], caixas_info[j][2], caixas_info[j][3]
            
            l_i_ef = l_i if r_i == 0 else w_i
            w_i_ef = w_i if r_i == 0 else l_i
            l_j_ef = l_j if r_j == 0 else w_j
            w_j_ef = w_j if r_j == 0 else l_j
            
            # Caixa i tem prioridade MAIOR (sai DEPOIS) que caixa j
            if d_i > d_j:
                # Verifica se caixa i está DEBAIXO da caixa j (i suporta j)
                if (z_i + h_i == z_j and  # i está imediatamente abaixo de j
                    x_i < x_j + l_j_ef and x_i + l_i_ef > x_j and  # sobreposição em x
                    y_i < y_j + w_j_ef and y_i + w_i_ef > y_j):    # sobreposição em y
                    # Penalidade: caixa que sai depois está debaixo de caixa que sai antes
                    penalidade += 20 * (d_i - d_j)
                
                # Verifica se caixa i está ATRÁS da caixa j (i bloqueada por j)
                elif (x_i > x_j and  # i está atrás de j (mais longe da porta)
                      z_i + h_i > z_j and z_i < z_j + h_j and  # sobreposição em z
                      y_i < y_j + w_j_ef and y_i + w_i_ef > y_j):  # sobreposição em y
                    # Penalidade: caixa que sai depois está bloqueada
                    penalidade += 10 * (d_i - d_j)
    
    return penalidade

def calcula_funcao_objetivo(posicoes, caixas_info, volume_total_container, alpha, beta, gamma):
    """Calcula a função objetivo f = alpha*U - beta*Ps - gamma*Pd"""
    
    # Volume utilizado
    volume_ocupado = calcula_volume_total_ocupado(posicoes, caixas_info)
    U = volume_ocupado / volume_total_container
    
    # Penalidades
    Ps = calcula_penalidade_instabilidade(posicoes, caixas_info)
    Pd = calcula_penalidade_ordem_entrega(posicoes, caixas_info)
    
    # Função objetivo
    f = alpha * U - beta * Ps - gamma * Pd
    
    return {
        'f': f,
        'U': U,
        'volume_ocupado': volume_ocupado,
        'Ps': Ps,
        'Pd': Pd
    }

def desenhar_caixa(ax, x, y, z, l, w, h, r, cor='blue', alpha=0.7, label=""):
    """Desenha uma caixa 3D"""
    
    l_ef = l if r == 0 else w
    w_ef = w if r == 0 else l
    h_ef = h
    
    vertices = np.array([
        [x, y, z],
        [x + l_ef, y, z],
        [x + l_ef, y + w_ef, z],
        [x, y + w_ef, z],
        [x, y, z + h_ef],
        [x + l_ef, y, z + h_ef],
        [x + l_ef, y + w_ef, z + h_ef],
        [x, y + w_ef, z + h_ef]
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]
    
    collection = Poly3DCollection(faces, alpha=alpha, facecolor=cor, edgecolor='black', linewidth=1)
    ax.add_collection3d(collection)
    
    # Adicionar texto com identificador
    ax.text(x + l_ef/2, y + w_ef/2, z + h_ef/2, label, 
            fontsize=8, color='black', weight='bold', ha='center', va='center')

def visualizar_e_salvar(solucao, id_solucao, pasta_imagens, caixas_info, L, W, H):
    """Visualiza uma solução e salva a imagem"""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Desenhar contêiner
    container_vertices = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
    ])
    
    edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
    for edge in edges:
        ax.plot3D(*zip(container_vertices[edge[0]], container_vertices[edge[1]]), 
                  color='black', linewidth=2, linestyle='--', alpha=0.5)
    
    # Cores
    cores = ['lightblue', 'lightgreen', 'salmon', 'orange', 'purple', 'pink', 'cyan', 'yellow']
    
    # Desenhar caixas
    for i, (pos, caixa) in enumerate(zip(solucao['posicoes'], caixas_info)):
        x, y, z, r = pos
        _, l, w, h, d, nome = caixa
        desenhar_caixa(ax, x, y, z, l, w, h, r, cores[i % len(cores)], alpha=0.8, label=f"B{i+1}\nd={d}")
    
    # Configurar gráfico
    ax.set_xlabel('X (comprimento) - Porta em X=0', fontsize=10)
    ax.set_ylabel('Y (largura)', fontsize=10)
    ax.set_zlabel('Z (altura)', fontsize=10)
    ax.set_xlim([0, L])
    ax.set_ylim([0, W])
    ax.set_zlim([0, H])
    
    # Criar título com função objetivo
    titulo = f"Solução {id_solucao}\n"
    titulo += " | ".join([f"B{i+1}{solucao['posicoes'][i]}" for i in range(len(solucao['posicoes']))])
    titulo += f"\nf = {solucao['f_objetivo']['f']:.4f} | U = {solucao['f_objetivo']['U']:.4f} | Ps = {solucao['f_objetivo']['Ps']:.2f} | Pd = {solucao['f_objetivo']['Pd']:.2f}"
    ax.set_title(titulo, fontsize=9, weight='bold')
    
    # Ajustar ângulo de visão
    ax.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    
    # Salvar imagem
    nome_arquivo = f"solucao_{id_solucao:04d}_f{solucao['f_objetivo']['f']:.4f}.png"
    caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
    plt.savefig(caminho_completo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return caminho_completo

def gerar_todas_combinacoes(posicoes_por_caixa):
    """Gera todas as combinações de posições usando produto cartesiano"""
    return product(*posicoes_por_caixa)

def main():
    print("="*60)
    print("GERADOR DE SOLUÇÕES VÁLIDAS COM FUNÇÃO OBJETIVO")
    print("="*60)
    
    n_caixas = len(caixas)
    volume_total_container = L * W * H
    print(f"\nContêiner: {L}x{W}x{H} = {volume_total_container} unidades de volume")
    print(f"Número de caixas: {n_caixas}")
    print(f"Parâmetros: α={ALPHA}, β={BETA}, γ={GAMMA}")
    
    # Gerar posições
    print("\n1. Gerando posições possíveis...")
    posicoes_caixas = []
    for caixa in caixas:
        cid, l, w, h, d, nome = caixa
        posicoes = gerar_posicoes(l, w, h, L, W, H)
        posicoes_caixas.append(posicoes)
        print(f"   Caixa {cid} ({nome}): {len(posicoes)} posições")
    
    # Calcular total de combinações
    total_combinacoes = 1
    for p in posicoes_caixas:
        total_combinacoes *= len(p)
    
    print(f"\n2. Total de combinações brutas: {total_combinacoes:,}")
    print("   Testando combinações e calculando função objetivo...")
    
    # Testar todas combinações
    solucoes = []
    count = 0
    melhor_f = -float('inf')
    melhor_solucao = None
    
    for combinacao in gerar_todas_combinacoes(posicoes_caixas):
        count += 1
        
        # Progresso
        if count % 10000 == 0:
            percent = (count / total_combinacoes) * 100
            print(f"   Progresso: {count:,}/{total_combinacoes:,} ({percent:.2f}%)")
        
        # Verificar restrições
        if verifica_todas_sobreposicoes(combinacao, caixas):
            continue
        
        if not verifica_estabilidade_geral(combinacao, caixas):
            continue
        
        if not verifica_ordem_entrega(combinacao, caixas):
            continue
        
        # Calcular função objetivo
        f_obj = calcula_funcao_objetivo(combinacao, caixas, volume_total_container, ALPHA, BETA, GAMMA)
        
        # Solução válida
        solucao = {
            'id': len(solucoes) + 1,
            'posicoes': list(combinacao),
            'descricao': " | ".join([f"B{i+1}{combinacao[i]}" for i in range(n_caixas)]),
            'f_objetivo': f_obj
        }
        solucoes.append(solucao)
        
        # Atualizar melhor solução
        if f_obj['f'] > melhor_f:
            melhor_f = f_obj['f']
            melhor_solucao = solucao
    
    print(f"\n   Total de soluções válidas: {len(solucoes):,}")
    
    # Ordenar soluções por função objetivo (melhores primeiro)
    solucoes.sort(key=lambda x: x['f_objetivo']['f'], reverse=True)
    
    # Reatribuir IDs após ordenação
    for i, sol in enumerate(solucoes):
        sol['id'] = i + 1
    
    # Salvar dados
    print("\n3. Salvando dados...")
    
    # Salvar JSON
    with open(f"{PASTA_SOLUCOES}/dados/todas_solucoes.json", "w", encoding="utf-8") as f:
        json.dump(solucoes, f, indent=2, ensure_ascii=False)
    
    # Salvar TXT
    with open(f"{PASTA_SOLUCOES}/dados/todas_solucoes.txt", "w", encoding="utf-8") as f:
        f.write("TODAS AS SOLUÇÕES VÁLIDAS (ORDENADAS POR FUNÇÃO OBJETIVO)\n")
        f.write("="*80 + "\n")
        f.write(f"Contêiner: {L}x{W}x{H} = {volume_total_container}\n")
        f.write(f"Caixas: {len(caixas)}\n")
        f.write(f"Parâmetros: α={ALPHA}, β={BETA}, γ={GAMMA}\n")
        f.write(f"Total: {len(solucoes)} soluções\n\n")
        f.write(f"{'ID':>4} {'f':>12} {'U':>8} {'Ps':>10} {'Pd':>10}  {'Descrição'}\n")
        f.write("-"*80 + "\n")
        for sol in solucoes:
            f.write(f"{sol['id']:4d} {sol['f_objetivo']['f']:12.4f} {sol['f_objetivo']['U']:8.4f} "
                   f"{sol['f_objetivo']['Ps']:10.2f} {sol['f_objetivo']['Pd']:10.2f}  {sol['descricao']}\n")
    
    # Salvar CSV
    with open(f"{PASTA_SOLUCOES}/dados/todas_solucoes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Cabeçalho dinâmico
        header = ["id", "f", "U", "volume_ocupado", "Ps", "Pd"]
        for i in range(n_caixas):
            header.extend([f"b{i+1}_x", f"b{i+1}_y", f"b{i+1}_z", f"b{i+1}_r"])
        writer.writerow(header)
        
        for sol in solucoes:
            row = [sol['id'], sol['f_objetivo']['f'], sol['f_objetivo']['U'], 
                   sol['f_objetivo']['volume_ocupado'], sol['f_objetivo']['Ps'], sol['f_objetivo']['Pd']]
            for pos in sol['posicoes']:
                row.extend([pos[0], pos[1], pos[2], pos[3]])
            writer.writerow(row)
    
    # Gerar imagens (apenas das melhores soluções)
    print("\n4. Gerando imagens...")
    pasta_imagens = f"{PASTA_SOLUCOES}/imagens"
    
    # Gerar imagens para todas as soluções (ou limitar a 100)
    max_imagens = min(len(solucoes), 100)
    for i in range(max_imagens):
        sol = solucoes[i]
        if (i + 1) % 10 == 0:
            print(f"   Gerando imagem {i+1}/{max_imagens}...")
        
        caminho = visualizar_e_salvar(sol, sol['id'], pasta_imagens, caixas, L, W, H)
        sol['caminho_imagem'] = caminho
    
    # Salvar JSON atualizado com caminhos das imagens
    with open(f"{PASTA_SOLUCOES}/dados/todas_solucoes_com_imagens.json", "w", encoding="utf-8") as f:
        json.dump(solucoes[:max_imagens], f, indent=2, ensure_ascii=False)
    
    # Relatório final
    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print("="*60)
    print(f"Total de soluções válidas: {len(solucoes):,}")
    print(f"\nMELHOR SOLUÇÃO:")
    melhor = solucoes[0]
    print(f"  ID: {melhor['id']}")
    print(f"  f = {melhor['f_objetivo']['f']:.4f}")
    print(f"  U = {melhor['f_objetivo']['U']:.4f} (volume: {melhor['f_objetivo']['volume_ocupado']}/{volume_total_container})")
    print(f"  Ps = {melhor['f_objetivo']['Ps']:.2f}")
    print(f"  Pd = {melhor['f_objetivo']['Pd']:.2f}")
    print(f"  Descrição: {melhor['descricao']}")
    
    print(f"\nPIOR SOLUÇÃO:")
    pior = solucoes[-1]
    print(f"  ID: {pior['id']}")
    print(f"  f = {pior['f_objetivo']['f']:.4f}")
    print(f"  U = {pior['f_objetivo']['U']:.4f}")
    print(f"  Ps = {pior['f_objetivo']['Ps']:.2f}")
    print(f"  Pd = {pior['f_objetivo']['Pd']:.2f}")
    
    print(f"\nArquivos salvos em: {PASTA_SOLUCOES}/")
    print(f"  - dados/todas_solucoes.json")
    print(f"  - dados/todas_solucoes.txt") 
    print(f"  - dados/todas_solucoes.csv")
    print(f"  - imagens/solucao_*.png ({max_imagens} imagens)")
    
    return solucoes

# EXECUTAR
if __name__ == "__main__":
    solucoes = main()