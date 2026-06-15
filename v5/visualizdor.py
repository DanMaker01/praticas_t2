"""
Visualizador 3D para soluções de empilhamento de caixas
Com garantia de que todos os cubos são visíveis
"""

import json
import os
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from datetime import datetime
from collections import defaultdict

class Visualizador3D:
    """Visualizador 3D para soluções de caixas em container"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.cores = plt.cm.tab20
    
    def carregar_resultados(self, caminho_pasta):
        """Carrega resultados de uma pasta de execução"""
        
        arquivo_json = os.path.join(caminho_pasta, 'solucao_completa.json')
        
        if not os.path.exists(arquivo_json):
            print(f"Arquivo nao encontrado: {arquivo_json}")
            return None
        
        with open(arquivo_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        print(f"Carregado: {caminho_pasta}")
        print(f"   Data: {dados['metadata']['data_execucao']}")
        print(f"   Fitness: {dados['metadata']['melhor_fitness']:.12f}")
        print(f"   Utilizacao: {dados['metadata']['utilizacao']:.1%}")
        print(f"   Caixas: {len(dados['caixas'])}")
        
        return dados
    
    def listar_execucoes(self, pasta_base="resultados_brkga"):
        """Lista todas as execuções disponíveis"""
        
        if not os.path.exists(pasta_base):
            print(f"Pasta nao encontrada: {pasta_base}")
            return []
        
        execucoes = []
        for item in os.listdir(pasta_base):
            caminho = os.path.join(pasta_base, item)
            if os.path.isdir(caminho) and item.startswith('execucao_'):
                arquivo_json = os.path.join(caminho, 'solucao_completa.json')
                if os.path.exists(arquivo_json):
                    with open(arquivo_json, 'r', encoding='utf-8') as f:
                        dados = json.load(f)
                    
                    execucoes.append({
                        'pasta': caminho,
                        'timestamp': item.replace('execucao_', ''),
                        'fitness': dados['metadata']['melhor_fitness'],
                        'utilizacao': dados['metadata']['utilizacao'],
                        'caixas': len(dados['caixas'])
                    })
        
        execucoes.sort(key=lambda x: x['timestamp'], reverse=True)
        return execucoes
    
    def extrair_posicoes(self, dados):
        """Extrai posições das caixas dos dados"""
        
        L = dados['parametros']['L']
        W = dados['parametros']['W']
        H = dados['parametros']['H']
        caixas = dados['caixas']
        solucao = dados['melhor_solucao']
        
        posicoes = []
        for i, pos in enumerate(solucao):
            if pos is None:
                continue
            
            x, y, z, rotacao = pos
            caixa = caixas[i]
            
            if rotacao == 0:
                l, w, h = caixa['l'], caixa['w'], caixa['h']
            else:
                l, w, h = caixa['w'], caixa['l'], caixa['h']
            
            posicoes.append({
                'indice': i,
                'nome': caixa['nome'],
                'prioridade': caixa['prioridade'],
                'x': x, 'y': y, 'z': z,
                'l': l, 'w': w, 'h': h,
                'rotacao': rotacao
            })
        
        return posicoes, L, W, H
    
    def desenhar_caixa(self, x, y, z, l, w, h, cor, alpha=0.9, borda=True):
        """Desenha uma caixa 3D com todas as faces visiveis"""
        
        vertices = np.array([
            [x, y, z],
            [x + l, y, z],
            [x + l, y + w, z],
            [x, y + w, z],
            [x, y, z + h],
            [x + l, y, z + h],
            [x + l, y + w, z + h],
            [x, y + w, z + h]
        ])
        
        # Todas as 6 faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # base
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # topo
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # frente
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # tras
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # esquerda
            [vertices[1], vertices[2], vertices[6], vertices[5]]    # direita
        ]
        
        # Desenhar todas as faces com opacidade alta
        collection = Poly3DCollection(faces, alpha=alpha, facecolor=cor, 
                                     edgecolor='black' if borda else None, 
                                     linewidth=0.8)
        self.ax.add_collection3d(collection)
    
    def desenhar_container_transparente(self, L, W, H):
        """Desenha o container com linhas semi-transparentes para nao esconder caixas"""
        
        vertices = np.array([
            [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
            [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
        ])
        
        arestas = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for aresta in arestas:
            self.ax.plot3D(
                [vertices[aresta[0]][0], vertices[aresta[1]][0]],
                [vertices[aresta[0]][1], vertices[aresta[1]][1]],
                [vertices[aresta[0]][2], vertices[aresta[1]][2]],
                'gray', linewidth=1.5, alpha=0.4
            )
    
    def validar_restricoes(self, posicoes, L, W, H):
        """Valida restrições para um conjunto de posições"""
        
        resultados = {
            'limites': {'ok': True, 'erros': []},
            'sobreposicao': {'ok': True, 'erros': []},
            'suporte': {'ok': True, 'erros': []},
            'ordem_x': {'ok': True, 'erros': []}
        }
        
        ocupado = set()
        for p in posicoes:
            for dx in range(p['l']):
                for dy in range(p['w']):
                    for dz in range(p['h']):
                        ocupado.add((p['x'] + dx, p['y'] + dy, p['z'] + dz))
        
        for p in posicoes:
            if p['x'] + p['l'] > L:
                resultados['limites']['erros'].append(f"{p['nome']} ultrapassa X")
            if p['y'] + p['w'] > W:
                resultados['limites']['erros'].append(f"{p['nome']} ultrapassa Y")
            if p['z'] + p['h'] > H:
                resultados['limites']['erros'].append(f"{p['nome']} ultrapassa Z")
        
        resultados['limites']['ok'] = len(resultados['limites']['erros']) == 0
        
        for i, p1 in enumerate(posicoes):
            for j, p2 in enumerate(posicoes):
                if i >= j:
                    continue
                
                x_overlap = (p1['x'] < p2['x'] + p2['l']) and (p2['x'] < p1['x'] + p1['l'])
                y_overlap = (p1['y'] < p2['y'] + p2['w']) and (p2['y'] < p1['y'] + p1['w'])
                z_overlap = (p1['z'] < p2['z'] + p2['h']) and (p2['z'] < p1['z'] + p1['h'])
                
                if x_overlap and y_overlap and z_overlap:
                    resultados['sobreposicao']['erros'].append(f"{p1['nome']} vs {p2['nome']}")
        
        resultados['sobreposicao']['ok'] = len(resultados['sobreposicao']['erros']) == 0
        
        for p in posicoes:
            if p['z'] == 0:
                continue
            
            area_total = p['l'] * p['w']
            area_apoiada = 0
            
            for dx in range(p['l']):
                for dy in range(p['w']):
                    if (p['x'] + dx, p['y'] + dy, p['z'] - 1) in ocupado:
                        area_apoiada += 1
            
            if area_apoiada < area_total:
                resultados['suporte']['erros'].append(f"{p['nome']}: {area_apoiada/area_total*100:.0f}% apoio")
        
        resultados['suporte']['ok'] = len(resultados['suporte']['erros']) == 0
        
        for i, p1 in enumerate(posicoes):
            for j, p2 in enumerate(posicoes):
                if i == j:
                    continue
                
                y_overlap = (p1['y'] < p2['y'] + p2['w']) and (p2['y'] < p1['y'] + p1['w'])
                z_overlap = (p1['z'] < p2['z'] + p2['h']) and (p2['z'] < p1['z'] + p1['h'])
                
                if y_overlap and z_overlap:
                    if p1['prioridade'] < p2['prioridade'] and p1['x'] > p2['x']:
                        resultados['ordem_x']['erros'].append(f"{p1['nome']}(P{p1['prioridade']}) atras de {p2['nome']}(P{p2['prioridade']})")
        
        resultados['ordem_x']['ok'] = len(resultados['ordem_x']['erros']) == 0
        
        return resultados
    
    def desenhar_cenario(self, posicoes, L, W, H, titulo, mostrar_infos=True):
        """Desenha um cenario completo com todas as caixas visiveis"""
        
        self.ax.clear()
        self.desenhar_container_transparente(L, W, H)
        
        prioridades = sorted(set(p['prioridade'] for p in posicoes))
        cores_prioridade = {p: self.cores(i % 20) for i, p in enumerate(prioridades)}
        
        # Desenhar todas as caixas sem ordem especial para garantir visibilidade
        for p in posicoes:
            cor = cores_prioridade[p['prioridade']]
            self.desenhar_caixa(p['x'], p['y'], p['z'], p['l'], p['w'], p['h'], cor, alpha=0.85, borda=True)
            
            if mostrar_infos:
                self.ax.text(p['x'] + p['l']/2, p['y'] + p['w']/2, p['z'] + p['h'] + 0.2,
                            f"{p['nome']}\nP{p['prioridade']}", 
                            ha='center', va='bottom', fontsize=7, alpha=0.9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        self.ax.set_xlabel('X (Profundidade)', fontsize=10)
        self.ax.set_ylabel('Y (Largura)', fontsize=10)
        self.ax.set_zlabel('Z (Altura)', fontsize=10)
        
        self.ax.set_xlim([0, L])
        self.ax.set_ylim([0, W])
        self.ax.set_zlim([0, H])
        
        self.ax.set_title(titulo, fontsize=12)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cores_prioridade[p], label=f'Prioridade {p}') 
                          for p in prioridades]
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Angulo que mostra melhor as caixas internas
        self.ax.view_init(elev=20, azim=112)
    
    def visualizar_descarga(self, dados):
        """Visualiza o processo de descarga passo a passo"""
        
        posicoes, L, W, H = self.extrair_posicoes(dados)
        
        caixas_por_prioridade = defaultdict(list)
        for p in posicoes:
            caixas_por_prioridade[p['prioridade']].append(p)
        
        prioridades = sorted(caixas_por_prioridade.keys())
        
        todas_prioridades = sorted(set(p['prioridade'] for p in posicoes))
        cores_fixas = {p: self.cores(i % 20) for i, p in enumerate(todas_prioridades)}
        
        n_passos = len(prioridades) + 1
        n_cols = min(3, n_passos)
        n_rows = (n_passos + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        
        ax1 = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
        self.ax = ax1
        self.desenhar_cenario_com_cores_fixas(posicoes, L, W, H, "Estado inicial: Todas as caixas", cores_fixas, mostrar_infos=True)
        
        caixas_restantes = posicoes.copy()
        
        for idx, prioridade in enumerate(prioridades, start=2):
            caixas_restantes = [p for p in caixas_restantes if p['prioridade'] != prioridade]
            
            ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
            self.ax = ax
            
            caixas_removidas = len([p for p in posicoes if p['prioridade'] == prioridade])
            titulo = f"Apos remover prioridade {prioridade}\n({caixas_removidas} removidas, {len(caixas_restantes)} restantes)"
            
            self.desenhar_cenario_com_cores_fixas(caixas_restantes, L, W, H, titulo, cores_fixas, mostrar_infos=True)
        
        plt.suptitle("Processo de Descarga por Prioridade", fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def visualizar_descarga_comparativa(self, dados):
        """Visualiza descarga com verificacao de estabilidade"""
        
        posicoes, L, W, H = self.extrair_posicoes(dados)
        
        caixas_por_prioridade = defaultdict(list)
        for p in posicoes:
            caixas_por_prioridade[p['prioridade']].append(p)
        
        prioridades = sorted(caixas_por_prioridade.keys())
        
        todas_prioridades = sorted(set(p['prioridade'] for p in posicoes))
        cores_fixas = {p: self.cores(i % 20) for i, p in enumerate(todas_prioridades)}
        
        passos = []
        caixas_restantes = posicoes.copy()
        
        validacao = self.validar_restricoes(caixas_restantes, L, W, H)
        passos.append({
            'titulo': 'Estado inicial',
            'posicoes': caixas_restantes.copy(),
            'valido': all([validacao[k]['ok'] for k in validacao])
        })
        
        for prioridade in prioridades:
            caixas_restantes = [p for p in caixas_restantes if p['prioridade'] != prioridade]
            validacao = self.validar_restricoes(caixas_restantes, L, W, H)
            
            passos.append({
                'titulo': f'Prioridade {prioridade} removida',
                'posicoes': caixas_restantes.copy(),
                'valido': all([validacao[k]['ok'] for k in validacao])
            })
        
        n_passos = len(passos)
        n_cols = min(3, n_passos)
        n_rows = (n_passos + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        
        for idx, passo in enumerate(passos, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx, projection='3d')
            self.ax = ax
            
            status = "[OK] VALIDO" if passo['valido'] else "[!!] INVALIDO"
            titulo = f"{passo['titulo']}\n{status} ({len(passo['posicoes'])} caixas)"
            
            self.desenhar_cenario_com_cores_fixas(passo['posicoes'], L, W, H, titulo, cores_fixas, mostrar_infos=True)
        
        plt.suptitle("Processo de Descarga com Validacao de Estabilidade", fontsize=14)
        plt.tight_layout()
        
        return fig

    def desenhar_cenario_com_cores_fixas(self, posicoes, L, W, H, titulo, cores_fixas, mostrar_infos=True):
        """Desenha cenario usando mapa de cores fixo"""
        
        self.ax.clear()
        self.desenhar_container_transparente(L, W, H)
        
        for p in posicoes:
            cor = cores_fixas[p['prioridade']]
            self.desenhar_caixa(p['x'], p['y'], p['z'], p['l'], p['w'], p['h'], cor, alpha=0.85, borda=True)
            
            if mostrar_infos:
                self.ax.text(p['x'] + p['l']/2, p['y'] + p['w']/2, p['z'] + p['h'] + 0.2,
                            f"{p['nome']}\nP{p['prioridade']}", 
                            ha='center', va='bottom', fontsize=7, alpha=0.9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        self.ax.set_xlabel('X (Profundidade)', fontsize=10)
        self.ax.set_ylabel('Y (Largura)', fontsize=10)
        self.ax.set_zlabel('Z (Altura)', fontsize=10)
        
        self.ax.set_xlim([0, L])
        self.ax.set_ylim([0, W])
        self.ax.set_zlim([0, H])
        
        self.ax.set_title(titulo, fontsize=11)
        
        from matplotlib.patches import Patch
        prioridades_restantes = sorted(set(p['prioridade'] for p in posicoes))
        legend_elements = [Patch(facecolor=cores_fixas[p], label=f'Prioridade {p}') 
                        for p in prioridades_restantes]
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        self.ax.view_init(elev=20, azim=112)
    
    def interativo(self):
        """Modo interativo para selecionar execucao"""
        
        print("\n" + "="*50)
        print("VISUALIZADOR 3D - SOLUCOES DE EMPILHAMENTO")
        print("="*50)
        
        execucoes = self.listar_execucoes()
        
        if not execucoes:
            print("Nenhuma execucao encontrada!")
            return
        
        print("\nExecucoes disponiveis:\n")
        for i, exec in enumerate(execucoes):
            print(f"{i+1}. {exec['timestamp']}")
            print(f"   Fitness: {exec['fitness']:.12f} | Utilizacao: {exec['utilizacao']:.1%} | Caixas: {exec['caixas']}")
            print()
        
        while True:
            try:
                escolha = input(f"Selecione uma execucao (1-{len(execucoes)}) ou 'q' para sair: ").strip()
                if escolha.lower() == 'q':
                    return
                
                idx = int(escolha) - 1
                if 0 <= idx < len(execucoes):
                    break
                print(f"Escolha invalida! Digite 1-{len(execucoes)}")
            except ValueError:
                print("Digite um numero valido!")
        
        exec_selecionada = execucoes[idx]
        dados = self.carregar_resultados(exec_selecionada['pasta'])
        
        if dados is None:
            return
        
        while True:
            print("\n" + "-"*30)
            print("Opcoes de visualizacao:")
            print("1 - Visualizacao 3D (com prioridades)")
            print("2 - Visualizacao 3D (sem prioridades)")
            print("3 - Visualizacao por camadas")
            print("4 - Evolucao do fitness")
            print("5 - DESCARGA PASSO A PASSO")
            print("6 - DESCARGA COM VALIDACAO")
            print("7 - Todas as visualizacoes")
            print("8 - Voltar")
            
            opcao = input("\nEscolha: ").strip()
            
            if opcao == '1':
                posicoes, L, W, H = self.extrair_posicoes(dados)
                fig = plt.figure(figsize=(14, 10))
                self.ax = fig.add_subplot(111, projection='3d')
                self.desenhar_cenario(posicoes, L, W, H, f"Visualizacao 3D - Fitness: {dados['metadata']['melhor_fitness']:.12f}", mostrar_infos=True)
                plt.show()
            elif opcao == '2':
                posicoes, L, W, H = self.extrair_posicoes(dados)
                fig = plt.figure(figsize=(14, 10))
                self.ax = fig.add_subplot(111, projection='3d')
                self.desenhar_cenario(posicoes, L, W, H, f"Visualizacao 3D - Fitness: {dados['metadata']['melhor_fitness']:.12f}", mostrar_infos=False)
                plt.show()
            elif opcao == '3':
                self.visualizar_camadas(dados)
                plt.show()
            elif opcao == '4':
                self.visualizar_evolucao(dados)
                plt.show()
            elif opcao == '5':
                fig = self.visualizar_descarga(dados)
                plt.show()
            elif opcao == '6':
                fig = self.visualizar_descarga_comparativa(dados)
                plt.show()
            elif opcao == '7':
                self.visualizar_descarga(dados)
                self.visualizar_descarga_comparativa(dados)
                self.visualizar_evolucao(dados)
                plt.show()
            elif opcao == '8':
                break
            else:
                print("Opcao invalida!")
    
    def visualizar_camadas(self, dados):
        """Visualiza camadas (slices) do container"""
        
        posicoes, L, W, H = self.extrair_posicoes(dados)
        
        alturas = np.arange(0, H + 1, max(1, H // 5))
        
        n_cols = min(4, len(alturas))
        n_rows = (len(alturas) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, z_corte in enumerate(alturas):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ax.set_xlim(0, L)
            ax.set_ylim(0, W)
            ax.set_title(f'Camada z = {z_corte:.0f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            for p in posicoes:
                if p['z'] <= z_corte < p['z'] + p['h']:
                    rect = plt.Rectangle((p['x'], p['y']), p['l'], p['w'],
                                        facecolor=self.cores(p['prioridade'] % 20),
                                        alpha=0.7, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(p['x'] + p['l']/2, p['y'] + p['w']/2,
                           f"{p['nome']}\nP{p['prioridade']}",
                           ha='center', va='center', fontsize=6)
        
        for idx in range(len(alturas), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Visualizacao por Camadas (Vista Superior)", fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def visualizar_evolucao(self, dados):
        """Visualiza evolucao do fitness ao longo das geracoes"""
        
        historico = dados['historico_fitness']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        geracoes = range(len(historico))
        ax.plot(geracoes, historico, 'b-', linewidth=2, label='Melhor Fitness')
        ax.fill_between(geracoes, historico, alpha=0.3)
        
        ax.set_xlabel('Geracao', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Evolucao do Fitness', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        melhor_idx = np.argmax(historico)
        ax.plot(melhor_idx, historico[melhor_idx], 'ro', markersize=8,
               label=f'Melhor: {historico[melhor_idx]:.12f}')
        
        return fig


def main():
    if len(sys.argv) > 1:
        caminho = sys.argv[1]
        visualizador = Visualizador3D()
        dados = visualizador.carregar_resultados(caminho)
        
        if dados:
            print("\n" + "-"*30)
            print("Gerando visualizacoes...")
            
            visualizador.visualizar_descarga(dados)
            visualizador.visualizar_descarga_comparativa(dados)
            plt.show()
    else:
        visualizador = Visualizador3D()
        visualizador.interativo()


if __name__ == "__main__":
    main()