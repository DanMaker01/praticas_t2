import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, CheckButtons
import re
import json
import os

class Visualizador3D:
    def __init__(self, solucao, caixas, L, W, H, fitness=None):
        self.solucao = solucao
        self.caixas = caixas
        self.L, self.W, self.H = L, W, H
        self.fitness = fitness
        self.visivel_individual = [True] * len(caixas)
        
        # Estado dos filtros por prioridade (True = ocultar)
        self.ocultar_prioridade = {}
        for p in set([c[4] for c in caixas]):
            self.ocultar_prioridade[p] = False
        
        # Mapeamento de labels para prioridades
        self.label_to_prioridade = {}
        
        self.cores = ['#FF4444', '#FF8844', '#FFCC44', '#88CC44', '#44CC88', '#44CCFF', '#4488FF', '#8844FF']
        
        # Criar figura
        self.fig = plt.figure(figsize=(15, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Desenhar
        self.desenhar_container()
        self.desenhar_caixas()
        self.configurar_eixos()
        self.criar_controles()
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def desenhar_container(self):
        v = np.array([[0,0,0],[self.L,0,0],[self.L,self.W,0],[0,self.W,0],
                      [0,0,self.H],[self.L,0,self.H],[self.L,self.W,self.H],[0,self.W,self.H]])
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        for e in edges:
            self.ax.plot3D(*zip(v[e[0]], v[e[1]]), color='black', linewidth=2)
    
    def desenhar_caixas(self):
        self.caixas_solid = []
        self.linhas = []
        self.textos = []
        
        for i, (pos, c) in enumerate(zip(self.solucao, self.caixas)):
            x, y, z, r = pos
            _, l, w, h, p, nome = c
            l_ef = l if r == 0 else w
            w_ef = w if r == 0 else l
            
            verts = np.array([
                [x,y,z], [x+l_ef,y,z], [x+l_ef,y+w_ef,z], [x,y+w_ef,z],
                [x,y,z+h], [x+l_ef,y,z+h], [x+l_ef,y+w_ef,z+h], [x,y+w_ef,z+h]
            ])
            
            faces = [
                [verts[0], verts[1], verts[2], verts[3]],
                [verts[4], verts[5], verts[6], verts[7]],
                [verts[0], verts[1], verts[5], verts[4]],
                [verts[2], verts[3], verts[7], verts[6]],
                [verts[1], verts[2], verts[6], verts[5]],
                [verts[0], verts[3], verts[7], verts[4]]
            ]
            
            cor = self.cores[p % len(self.cores)]
            poly = Poly3DCollection(faces, alpha=0.7, facecolor=cor, edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
            self.caixas_solid.append(poly)
            
            edges_caixa = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            lines = []
            for e in edges_caixa:
                line, = self.ax.plot3D(*zip(verts[e[0]], verts[e[1]]), color='black', linewidth=1, alpha=0.5)
                lines.append(line)
            self.linhas.append(lines)
            
            text = self.ax.text(x+l_ef/2, y+w_ef/2, z+h/2, f'B{i+1}\nP={p}',
                              fontsize=8, ha='center', va='center',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=cor, linewidth=2))
            self.textos.append(text)
        
        self.atualizar_visibilidade()
    
    def atualizar_visibilidade(self):
        for i in range(len(self.caixas)):
            p = self.caixas[i][4]
            
            # Verifica se a caixa deve ser visível
            if not self.visivel_individual[i]:
                vis = False
            elif self.ocultar_prioridade.get(p, False):
                vis = False
            else:
                vis = True
            
            alpha = 0.7 if vis else 0.1
            
            self.caixas_solid[i].set_alpha(alpha)
            self.caixas_solid[i].set_visible(vis)
            
            for line in self.linhas[i]:
                line.set_alpha(alpha if vis else 0)
                line.set_visible(vis)
            
            self.textos[i].set_visible(vis)
        
        self.fig.canvas.draw_idle()
    
    def configurar_eixos(self):
        self.ax.set_xlabel('X (Porta)')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.W)
        self.ax.set_zlim(0, self.H)
        self.ax.view_init(elev=25, azim=-45)
    
    def criar_controles(self):
        plt.subplots_adjust(right=0.72)
        
        # Botões
        ax_reset = plt.axes([0.74, 0.92, 0.24, 0.05])
        btn_reset = Button(ax_reset, 'Reset View', color='#4CAF50')
        btn_reset.on_clicked(self.reset_view)
        
        ax_show = plt.axes([0.74, 0.86, 0.11, 0.05])
        btn_show = Button(ax_show, 'Mostrar Tudo', color='#2196F3')
        btn_show.on_clicked(self.show_all)
        
        ax_hide = plt.axes([0.87, 0.86, 0.11, 0.05])
        btn_hide = Button(ax_hide, 'Esconder Tudo', color='#f44336')
        btn_hide.on_clicked(self.hide_all)
        
        # Filtro por prioridade
        ax_prior = plt.axes([0.74, 0.68, 0.24, 0.16])
        ax_prior.set_title('Ocultar por Prioridade', fontsize=11, fontweight='bold')
        
        prioridades = sorted(self.ocultar_prioridade.keys())
        labels_prior = []
        for p in prioridades:
            if p == 0:
                labels_prior.append(f'Prioridade {p} (sai primeiro)')
            else:
                labels_prior.append(f'Prioridade {p}')
            self.label_to_prioridade[labels_prior[-1]] = p
        
        self.check_prior = CheckButtons(ax_prior, labels_prior, [False] * len(prioridades))
        self.check_prior.on_clicked(self.toggle_prioridade)
        
        # Lista de caixas individuais
        ax_ind = plt.axes([0.74, 0.15, 0.24, 0.48])
        ax_ind.set_title('Ocultar Individualmente', fontsize=11, fontweight='bold')
        labels_ind = [f"B{i+1} | {c[1]}x{c[2]}x{c[3]} | P={c[4]}" for i, c in enumerate(self.caixas)]
        self.check_ind = CheckButtons(ax_ind, labels_ind, self.visivel_individual)
        self.check_ind.on_clicked(self.toggle_caixa)
        
        # Informações
        vol_total = self.L * self.W * self.H
        vol_ocupado = sum([c[1]*c[2]*c[3] for c in self.caixas])
        info = f"Fitness: {self.fitness:.4f}\n" if self.fitness else ""
        info += f"Volume: {vol_ocupado}/{vol_total}\nUso: {vol_ocupado/vol_total:.1%}\n"
        info += "Atalhos: R (reset), 1-9 (toggle)\n"
        info += "Filtros: clique nas prioridades ou caixas"
        self.ax.text2D(0.02, 0.98, info, transform=self.ax.transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    def toggle_caixa(self, label):
        """Alterna visibilidade de uma caixa individual"""
        idx = int(label.split()[0][1:]) - 1
        self.visivel_individual[idx] = not self.visivel_individual[idx]
        self.atualizar_visibilidade()
        print(f"Caixa {idx+1}: {'OCULTA' if not self.visivel_individual[idx] else 'VISÍVEL'}")
    
    def toggle_prioridade(self, label):
        """Alterna filtro por prioridade"""
        # Extrair a prioridade do label
        p = self.label_to_prioridade.get(label)
        if p is not None:
            # Inverte o estado
            self.ocultar_prioridade[p] = not self.ocultar_prioridade[p]
            print(f"Prioridade {p}: {'OCULTAR' if self.ocultar_prioridade[p] else 'MOSTRAR'}")
            self.atualizar_visibilidade()
    
    def reset_view(self, event):
        self.ax.view_init(elev=25, azim=-45)
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.W)
        self.ax.set_zlim(0, self.H)
        self.fig.canvas.draw_idle()
    
    def show_all(self, event):
        """Mostra todas as caixas (remove todos os filtros)"""
        # Resetar visibilidade individual
        for i in range(len(self.visivel_individual)):
            self.visivel_individual[i] = True
            self.check_ind.set_active(i)
        
        # Resetar filtros de prioridade
        for p in self.ocultar_prioridade:
            self.ocultar_prioridade[p] = False
        
        # Desmarcar todos os checkboxes de prioridade
        for i in range(len(self.check_prior.labels)):
            self.check_prior.set_active(i)
        
        self.atualizar_visibilidade()
        print("Mostrar tudo: todos os filtros removidos")
    
    def hide_all(self, event):
        """Esconde todas as caixas"""
        for i in range(len(self.visivel_individual)):
            self.visivel_individual[i] = False
            self.check_ind.set_active(i)
        
        self.atualizar_visibilidade()
        print("Esconder tudo: todas as caixas ocultadas")
    
    def on_key_press(self, event):
        if event.key == 'r':
            self.reset_view(None)
        elif event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(self.visivel_individual):
                self.visivel_individual[idx] = not self.visivel_individual[idx]
                self.check_ind.set_active(idx)
                self.atualizar_visibilidade()
    
    def mostrar(self):
        plt.show()


def carregar_resultados(caminho="resultados_brkga/resultados.json"):
    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        return None
    with open(caminho, 'r', encoding="utf-8") as f:
        dados = json.load(f)
    solucao = [tuple(pos) for pos in dados['melhor_solucao']]
    caixas = [(c['id'], c['l'], c['w'], c['h'], c['prioridade'], c['nome']) for c in dados['caixas']]
    return solucao, caixas, dados['L'], dados['W'], dados['H'], dados['melhor_fitness']