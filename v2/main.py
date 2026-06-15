import pygame
import sys
import math
import random
from typing import List, Tuple, Dict, Set
import numpy as np
from collections import defaultdict
import threading
import time

# ==================== CONFIGURAÇÕES ====================

# Cores
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
CINZA = (128, 128, 128)
CINZA_CLARO = (200, 200, 200)
CINZA_ESCURO = (64, 64, 64)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
AMARELO = (255, 255, 0)
VERMELHO_CLARO = (255, 100, 100)
AZUL_CLARO = (100, 100, 255)
VERDE_CLARO = (100, 255, 100)

# Cores por prioridade
CORES_PRIORIDADE = {
    1: VERMELHO,
    2: AZUL,
    3: VERDE
}

CORES_PRIORIDADE_CLARA = {
    1: VERMELHO_CLARO,
    2: AZUL_CLARO,
    3: VERDE_CLARO
}

# Dimensões da tela
LARGURA = 1400
ALTURA = 900

# Parâmetros da projeção isométrica
ISO_ANGULO = 30
ISO_COS = math.cos(math.radians(ISO_ANGULO))
ISO_SEN = math.sin(math.radians(ISO_ANGULO))
ESCALA = 3.5
OFFSET_X = LARGURA // 2 + 100
OFFSET_Y = ALTURA // 2 - 50

# ==================== CLASSES ====================

class Container:
    def __init__(self, x=100, y=50, z=20):
        self.X = x
        self.Y = y
        self.Z = z
        self.cor = CINZA_CLARO
    
    def projetar_isometrico(self, x, y, z):
        x2D = (x - y) * ISO_COS * ESCALA + OFFSET_X
        y2D = (x + y) * ISO_SEN * ESCALA - z * ESCALA + OFFSET_Y
        return (int(x2D), int(y2D))
    
    def desenhar(self, tela):
        # Face traseira (saída) - destaque
        p0 = self.projetar_isometrico(0, 0, 0)
        p1 = self.projetar_isometrico(self.X, 0, 0)
        p2 = self.projetar_isometrico(self.X, 0, self.Z)
        p3 = self.projetar_isometrico(0, 0, self.Z)
        pygame.draw.polygon(tela, AMARELO, [p0, p1, p2, p3], 3)
        
        # Outras faces
        for y in [0, self.Y]:
            for x in [0, self.X]:
                p0 = self.projetar_isometrico(x, y, 0)
                p1 = self.projetar_isometrico(x+self.X if x==0 else x, y, 0)
                p2 = self.projetar_isometrico(x+self.X if x==0 else x, y, self.Z)
                p3 = self.projetar_isometrico(x, y, self.Z)
                if y == self.Y or x == self.X:
                    pygame.draw.polygon(tela, CINZA_CLARO, [p0, p1, p2, p3], 1)

class Caixa:
    def __init__(self, tipo, dimensoes, prioridade, quantidade, id_unico=None):
        self.tipo = tipo
        self.dimensoes_base = dimensoes
        self.prioridade = prioridade
        self.quantidade = quantidade
        self.id_unico = id_unico
        self.posicao = None
        self.orientacao = 0
        
    def get_dimensoes(self, orientacao=None):
        if orientacao is None:
            orientacao = self.orientacao
        if orientacao == 0:
            return self.dimensoes_base
        else:
            return (self.dimensoes_base[1], self.dimensoes_base[0], self.dimensoes_base[2])
    
    def get_cor(self):
        return CORES_PRIORIDADE[self.prioridade]
    
    def set_posicao(self, x, y, z, orientacao):
        self.posicao = (x, y, z)
        self.orientacao = orientacao
    
    def desenhar(self, tela, container):
        if self.posicao is None:
            return
        
        x, y, z = self.posicao
        dx, dy, dz = self.get_dimensoes()
        
        cor = self.get_cor()
        cor_clara = tuple(min(255, c + 80) for c in cor)
        cor_escura = tuple(max(0, c - 80) for c in cor)
        
        # Desenhar faces
        faces = [
            ([x, x+dx, x+dx, x], [y, y, y+dy, y+dy], [z, z, z, z]),  # base
            ([x, x+dx, x+dx, x], [y, y, y+dy, y+dy], [z+dz, z+dz, z+dz, z+dz]),  # topo
            ([x, x, x+dx, x+dx], [y, y+dy, y+dy, y], [z, z, z, z]),  # frente
            ([x, x, x+dx, x+dx], [y, y+dy, y+dy, y], [z+dz, z+dz, z+dz, z+dz]),  # tras
            ([x, x+dx, x+dx, x], [y, y, y, y], [z, z, z+dz, z+dz]),  # esquerda
            ([x, x+dx, x+dx, x], [y+dy, y+dy, y+dy, y+dy], [z, z, z+dz, z+dz]),  # direita
        ]
        
        for i, (xs, ys, zs) in enumerate(faces):
            pontos = [container.projetar_isometrico(xs[j], ys[j], zs[j]) for j in range(4)]
            if i == 0:
                cor_face = cor_escura
            elif i == 1:
                cor_face = cor_clara
            else:
                cor_face = cor
            pygame.draw.polygon(tela, cor_face, pontos)
            pygame.draw.polygon(tela, PRETO, pontos, 2)
        
        # Texto
        centro = container.projetar_isometrico(x+dx/2, y+dy/2, z+dz/2)
        fonte = pygame.font.Font(None, 18)
        texto = fonte.render(f"{self.tipo}", True, BRANCO)
        texto_rect = texto.get_rect(center=centro)
        tela.blit(texto, texto_rect)

# ==================== GERENCIAMENTO DE ESPAÇO ====================

class VoxelSpace:
    def __init__(self, container):
        self.container = container
        self.ocupado = set()
        
    def pode_colocar(self, caixa, x, y, z, orient):
        dx, dy, dz = caixa.get_dimensoes(orient)
        
        if x < 0 or y < 0 or z < 0:
            return False
        if x + dx > self.container.X:
            return False
        if y + dy > self.container.Y:
            return False
        if z + dz > self.container.Z:
            return False
        
        for xi in range(x, x+dx):
            for yi in range(y, y+dy):
                for zi in range(z, z+dz):
                    if (xi, yi, zi) in self.ocupado:
                        return False
        return True
    
    def colocar(self, caixa, x, y, z, orient):
        dx, dy, dz = caixa.get_dimensoes(orient)
        for xi in range(x, x+dx):
            for yi in range(y, y+dy):
                for zi in range(z, z+dz):
                    self.ocupado.add((xi, yi, zi))

# ==================== BRKGA SIMPLIFICADO MAIS RÁPIDO ====================

class BRKGARapido:
    def __init__(self, demanda, container, pop_size=100, num_geracoes=200):
        self.demanda = demanda
        self.container = container
        self.num_caixas = len(demanda)
        self.pop_size = pop_size
        self.num_geracoes = num_geracoes
        self.populacao = None
        self.fitness = None
        self.melhor_fitness = 0
        self.melhor_solucao = None
        self.geracao_atual = 0
        
    def _gerar_solucao_gulosa(self):
        """Gera uma solução usando heurística gulosa"""
        posicoes = {}
        espaco = VoxelSpace(self.container)
        
        # Ordenar por prioridade (maior primeiro)
        indices = sorted(range(self.num_caixas), 
                        key=lambda i: self.demanda[i].prioridade)
        
        for i in indices:
            caixa = self.demanda[i]
            melhor_pos = None
            melhor_score = -1
            
            # Tentar orientações
            for orient in [0, 1]:
                dx, dy, dz = caixa.get_dimensoes(orient)
                
                # Buscar posição (heurística: cantos)
                for x in [0, self.container.X - dx]:
                    for y in [0, min(20, self.container.Y - dy)]:
                        for z in [0, self.container.Z - dz]:
                            if espaco.pode_colocar(caixa, x, y, z, orient):
                                score = 0
                                if caixa.prioridade == 1:
                                    score += (self.container.Y - y) * 100
                                score += (self.container.Z - z) * 10
                                if score > melhor_score:
                                    melhor_score = score
                                    melhor_pos = (x, y, z, orient)
            
            if melhor_pos:
                x, y, z, orient = melhor_pos
                espaco.colocar(caixa, x, y, z, orient)
                posicoes[i] = (x, y, z, orient)
        
        return posicoes
    
    def _calcular_fitness(self, posicoes):
        fitness = 0
        for caixa_id in posicoes:
            prioridade = self.demanda[caixa_id].prioridade
            if prioridade == 1:
                fitness += 1000
            elif prioridade == 2:
                fitness += 100
            else:
                fitness += 10
        return fitness
    
    def _mutacao(self, posicoes):
        """Aplica mutação trocando posições aleatórias"""
        if not posicoes or random.random() > 0.3:
            return posicoes
        
        nova_posicoes = posicoes.copy()
        
        # Remover algumas caixas aleatórias
        remover = random.sample(list(nova_posicoes.keys()), 
                               min(3, len(nova_posicoes)))
        for caixa_id in remover:
            del nova_posicoes[caixa_id]
        
        # Tentar recolocar em novas posições
        espaco = VoxelSpace(self.container)
        for caixa_id, pos in nova_posicoes.items():
            caixa = self.demanda[caixa_id]
            espaco.colocar(caixa, pos[0], pos[1], pos[2], pos[3])
        
        indices = sorted(remover, key=lambda i: self.demanda[i].prioridade)
        
        for i in indices:
            caixa = self.demanda[i]
            encontrou = False
            
            for orient in [0, 1]:
                dx, dy, dz = caixa.get_dimensoes(orient)
                for _ in range(50):  # Tentativas aleatórias
                    x = random.randint(0, self.container.X - dx)
                    y = random.randint(0, self.container.Y - dy)
                    z = random.randint(0, self.container.Z - dz)
                    if espaco.pode_colocar(caixa, x, y, z, orient):
                        espaco.colocar(caixa, x, y, z, orient)
                        nova_posicoes[i] = (x, y, z, orient)
                        encontrou = True
                        break
                if encontrou:
                    break
        
        return nova_posicoes
    
    def _cruzar(self, sol1, sol2):
        """Cruzamento entre duas soluções"""
        nova_sol = {}
        espaco = VoxelSpace(self.container)
        
        # Unir caixas de ambas soluções (prioridade)
        todas_caixas = set(sol1.keys()) | set(sol2.keys())
        
        for caixa_id in sorted(todas_caixas, 
                              key=lambda i: self.demanda[i].prioridade):
            if caixa_id in sol1 and caixa_id in sol2:
                # Escolher aleatoriamente
                sol = sol1 if random.random() < 0.5 else sol2
                x, y, z, orient = sol[caixa_id]
            elif caixa_id in sol1:
                x, y, z, orient = sol1[caixa_id]
            else:
                x, y, z, orient = sol2[caixa_id]
            
            caixa = self.demanda[caixa_id]
            if espaco.pode_colocar(caixa, x, y, z, orient):
                espaco.colocar(caixa, x, y, z, orient)
                nova_sol[caixa_id] = (x, y, z, orient)
        
        return nova_sol
    
    def executar(self, callback=None):
        """Executa o algoritmo"""
        # População inicial
        populacao = []
        for _ in range(self.pop_size):
            sol = self._gerar_solucao_gulosa()
            populacao.append(sol)
        
        # Avaliar
        fitness = [self._calcular_fitness(sol) for sol in populacao]
        
        for geracao in range(self.num_geracoes):
            self.geracao_atual = geracao
            
            # Ordenar
            indices = np.argsort(fitness)[::-1]
            
            # Melhor solução
            melhor_idx = indices[0]
            if fitness[melhor_idx] > self.melhor_fitness:
                self.melhor_fitness = fitness[melhor_idx]
                self.melhor_solucao = populacao[melhor_idx].copy()
            
            # Elite (top 20%)
            elite_size = self.pop_size // 5
            elite = [populacao[i] for i in indices[:elite_size]]
            
            # Cruzamento
            nova_populacao = elite.copy()
            
            while len(nova_populacao) < self.pop_size:
                pai1 = random.choice(elite)
                pai2 = random.choice(populacao)
                filho = self._cruzar(pai1, pai2)
                filho = self._mutacao(filho)
                nova_populacao.append(filho)
            
            populacao = nova_populacao
            fitness = [self._calcular_fitness(sol) for sol in populacao]
            
            if callback and geracao % 10 == 0:
                callback(geracao, self.melhor_fitness, len(self.melhor_solucao))
        
        return self.melhor_solucao

# ==================== DADOS DO PROBLEMA ====================

TIPOS_CAIXAS = [
    ('A', (10, 10, 20), 1, 3),
    ('B', (5, 20, 10), 1, 2),
    ('C', (10, 10, 20), 2, 1),
    ('D', (5, 5, 5), 2, 6),
    ('E', (3, 3, 20), 3, 1),
]

# Criar demanda
DEMANDA = []
id_counter = 0
for tipo, dim, prio, qtd in TIPOS_CAIXAS:
    for _ in range(qtd):
        DEMANDA.append(Caixa(tipo, dim, prio, qtd, id_counter))
        id_counter += 1

NUM_CAIXAS = len(DEMANDA)

# ==================== INTERFACE PYGAME ====================

class Game3D:
    def __init__(self):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption("Empacotamento 3D - BRKGA")
        self.clock = pygame.time.Clock()
        self.rodando = True
        self.container = Container(100, 50, 20)
        
        # Botões
        self.botoes = []
        self.fonte = pygame.font.Font(None, 24)
        self.fonte_grande = pygame.font.Font(None, 36)
        self.fonte_pequena = pygame.font.Font(None, 18)
        
        # Estado
        self.melhor_solucao = None
        self.executando = False
        self.thread = None
        self.progresso = 0
        self.melhor_fitness = 0
        self.geracao_atual = 0
        
        self.criar_botoes()
    
    def criar_botoes(self):
        botoes = [
            ("Executar BRKGA", 50, 750, self.executar_brkga),
            ("Limpar", 250, 750, self.limpar),
            ("Sair", 450, 750, self.sair),
        ]
        
        for texto, x, y, acao in botoes:
            rect = pygame.Rect(x, y, 180, 45)
            self.botoes.append({'rect': rect, 'texto': texto, 'acao': acao})
    
    def atualizar_progresso(self, geracao, fitness, num_caixas):
        """Callback para atualizar progresso"""
        self.geracao_atual = geracao
        self.melhor_fitness = fitness
        self.progresso = num_caixas
    
    def executar_brkga(self):
        if self.executando:
            return
        
        self.executando = True
        self.melhor_solucao = None
        self.progresso = 0
        
        def rodar_brkga():
            brkga = BRKGARapido(DEMANDA, self.container, 
                               pop_size=80, num_geracoes=150)
            solucao = brkga.executar(callback=self.atualizar_progresso)
            self.melhor_solucao = solucao
            
            # Atualizar visualização
            for caixa_id, (x, y, z, orient) in solucao.items():
                DEMANDA[caixa_id].set_posicao(x, y, z, orient)
            
            self.executando = False
        
        self.thread = threading.Thread(target=rodar_brkga)
        self.thread.daemon = True
        self.thread.start()
    
    def limpar(self):
        self.melhor_solucao = None
        for caixa in DEMANDA:
            caixa.posicao = None
        self.progresso = 0
        self.melhor_fitness = 0
        self.geracao_atual = 0
    
    def sair(self):
        self.rodando = False
    
    def desenhar(self):
        self.tela.fill(PRETO)
        self.container.desenhar(self.tela)
        
        # Desenhar caixas
        if self.melhor_solucao:
            for caixa_id, (x, y, z, orient) in self.melhor_solucao.items():
                caixa = DEMANDA[caixa_id]
                caixa.desenhar(self.tela, self.container)
        
        # Botões
        for botao in self.botoes:
            cor = CINZA_ESCURO if self.executando else CINZA
            pygame.draw.rect(self.tela, cor, botao['rect'])
            pygame.draw.rect(self.tela, BRANCO, botao['rect'], 2)
            texto = self.fonte.render(botao['texto'], True, BRANCO)
            texto_rect = texto.get_rect(center=botao['rect'].center)
            self.tela.blit(texto, texto_rect)
        
        # Status
        if self.executando:
            status = f"Executando... Geração {self.geracao_atual} | Fitness: {self.melhor_fitness:.0f}"
        elif self.melhor_solucao:
            status = f"Concluído! {len(self.melhor_solucao)}/{NUM_CAIXAS} caixas | Fitness: {self.melhor_fitness:.0f}"
        else:
            status = "Clique em Executar BRKGA"
        
        texto_status = self.fonte_grande.render(status, True, AMARELO)
        self.tela.blit(texto_status, (50, 50))
        
        # Barra de progresso (simulada)
        if self.executando:
            progresso = self.geracao_atual / 150
            barra_x = 50
            barra_y = 100
            barra_largura = 400
            barra_altura = 20
            pygame.draw.rect(self.tela, CINZA_ESCURO, (barra_x, barra_y, barra_largura, barra_altura))
            pygame.draw.rect(self.tela, VERDE, (barra_x, barra_y, int(barra_largura * progresso), barra_altura))
            texto_prog = self.fonte_pequena.render(f"{int(progresso * 100)}%", True, BRANCO)
            self.tela.blit(texto_prog, (barra_x + barra_largura//2 - 20, barra_y + 3))
        
        # Legenda de prioridades
        y = 150
        for prio, cor in [(1, VERMELHO), (2, AZUL), (3, VERDE)]:
            pygame.draw.rect(self.tela, cor, (50, y, 20, 20))
            texto = self.fonte.render(f"Prioridade {prio}", True, BRANCO)
            self.tela.blit(texto, (80, y))
            y += 30
        
        # Legenda de tipos de caixa
        y = 260
        for tipo, dim, prio, _ in TIPOS_CAIXAS:
            texto = self.fonte_pequena.render(f"{tipo}: {dim[0]}x{dim[1]}x{dim[2]} (P{prio})", True, CINZA_CLARO)
            self.tela.blit(texto, (50, y))
            y += 25
    
    def rodar(self):
        while self.rodando:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.rodando = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for botao in self.botoes:
                        if botao['rect'].collidepoint(pos) and not self.executando:
                            botao['acao']()
            
            self.desenhar()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("EMPACOTAMENTO 3D - BRKGA")
    print("=" * 60)
    print(f"\nContainer: 100 x 50 x 20")
    print(f"Total de caixas: {NUM_CAIXAS}")
    print("\nCaixas:")
    for tipo, dim, prio, qtd in TIPOS_CAIXAS:
        print(f"  Tipo {tipo}: {dim} - Prioridade {prio} - {qtd} unidades")
    
    print("\nExecutando interface...")
    game = Game3D()
    game.rodar()