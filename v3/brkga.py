import random
import numpy as np
import json
import os

class BRKGA:
    """
    Implementação do Biased Random-Key Genetic Algorithm para Empacotamento 3D
    """
    
    def __init__(self, num_genes, num_individuos, num_elite, num_mutantes, prob_crossover, 
                 caixas, L, W, H, alpha=1.0, beta=50.0, gamma=5.0, verbose=True):
        
        self.num_genes = num_genes
        self.num_individuos = num_individuos
        self.num_elite = num_elite
        self.num_mutantes = num_mutantes
        self.prob_crossover = prob_crossover
        self.caixas = caixas
        self.L = L
        self.W = W
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.verbose = verbose
        
        self.populacao = []
        self.fitness = []
        self.melhor_solucao = None
        self.melhor_fitness = -float('inf')
        self.historico_melhores = []
        
        self.pasta_resultados = "resultados_brkga"
        os.makedirs(self.pasta_resultados, exist_ok=True)
    
    def get_dimensoes_efetivas(self, caixa, r):
        """Retorna as dimensões efetivas (l, w, h) após rotação"""
        _, l, w, h, _, _ = caixa
        if r == 0:
            return l, w, h
        else:
            return w, l, h  # Rotação de 90° no plano XY (troca l com w)
    
    def sobrepoe(self, pos1, caixa1, pos2, caixa2):
        """Verifica se duas caixas se sobrepõem"""
        x1, y1, z1, r1 = pos1
        x2, y2, z2, r2 = pos2
        
        l1, w1, h1 = self.get_dimensoes_efetivas(caixa1, r1)
        l2, w2, h2 = self.get_dimensoes_efetivas(caixa2, r2)
        
        # Verifica sobreposição em cada eixo
        x_over = (x1 < x2 + l2 and x1 + l1 > x2)
        y_over = (y1 < y2 + w2 and y1 + w1 > y2)
        z_over = (z1 < z2 + h2 and z1 + h1 > z2)
        
        return x_over and y_over and z_over
    
    def verifica_todas_sobreposicoes(self, posicoes, debug=False):
        """Verifica se há qualquer sobreposição entre as caixas"""
        n = len(posicoes)
        for i in range(n):
            for j in range(i+1, n):
                if self.sobrepoe(posicoes[i], self.caixas[i], posicoes[j], self.caixas[j]):
                    if debug:
                        l1, w1, h1 = self.get_dimensoes_efetivas(self.caixas[i], posicoes[i][3])
                        l2, w2, h2 = self.get_dimensoes_efetivas(self.caixas[j], posicoes[j][3])
                        print(f"SOBREPOSIÇÃO entre caixa {i+1} e {j+1}")
                        print(f"  Caixa {i+1}: pos={posicoes[i]}, dim={l1}x{w1}x{h1}")
                        print(f"  Caixa {j+1}: pos={posicoes[j]}, dim={l2}x{w2}x{h2}")
                    return True
        return False
    
    def verifica_estabilidade_geral(self, posicoes):
        """Verifica se todas as caixas estão completamente apoiadas"""
        n = len(posicoes)
        for i in range(n):
            x, y, z, r = posicoes[i]
            if z == 0:
                continue
            
            l_ef, w_ef, h_ef = self.get_dimensoes_efetivas(self.caixas[i], r)
            
            # Verifica cada ponto da base da caixa
            for dx in range(l_ef):
                for dy in range(w_ef):
                    ponto_suportado = False
                    for j in range(n):
                        if i == j:
                            continue
                        xo, yo, zo, ro = posicoes[j]
                        lo_ef, wo_ef, ho_ef = self.get_dimensoes_efetivas(self.caixas[j], ro)
                        
                        # Verifica se o ponto (x+dx, y+dy) está apoiado na caixa j
                        if (xo <= x + dx < xo + lo_ef and 
                            yo <= y + dy < yo + wo_ef and 
                            zo + ho_ef == z):
                            ponto_suportado = True
                            break
                    
                    if not ponto_suportado:
                        return False
        return True
    
    def verifica_ordem_entrega(self, posicoes):
        """Verifica se a ordem de entrega é respeitada"""
        n = len(posicoes)
        for i in range(n):
            for j in range(i+1, n):
                d_i = self.caixas[i][4]
                d_j = self.caixas[j][4]
                x_i = posicoes[i][0]
                x_j = posicoes[j][0]
                
                if d_i < d_j and x_i > x_j:
                    return False
                if d_j < d_i and x_j > x_i:
                    return False
        return True
    
    def calcula_volume_ocupado(self, posicoes):
        """Calcula o volume total ocupado pelas caixas"""
        volume = 0
        for i, pos in enumerate(posicoes):
            _, l, w, h, _, _ = self.caixas[i]
            x, y, z, r = pos
            l_ef, w_ef, h_ef = self.get_dimensoes_efetivas(self.caixas[i], r)
            volume += l_ef * w_ef * h_ef
        return volume
    
    def calcula_penalidade_instabilidade(self, posicoes):
        """Calcula penalidade para caixas instáveis"""
        penalidade = 0
        n = len(posicoes)
        
        for i in range(n):
            x, y, z, r = posicoes[i]
            if z == 0:
                continue
            
            l_ef, w_ef, h_ef = self.get_dimensoes_efetivas(self.caixas[i], r)
            
            area_total = l_ef * w_ef
            area_apoiada = 0
            
            for dx in range(l_ef):
                for dy in range(w_ef):
                    for j in range(n):
                        if i == j:
                            continue
                        xo, yo, zo, ro = posicoes[j]
                        lo_ef, wo_ef, ho_ef = self.get_dimensoes_efetivas(self.caixas[j], ro)
                        
                        if (xo <= x + dx < xo + lo_ef and 
                            yo <= y + dy < yo + wo_ef and 
                            zo + ho_ef == z):
                            area_apoiada += 1
                            break
            
            area_nao_apoiada = area_total - area_apoiada
            if area_nao_apoiada > 0:
                penalidade += (area_nao_apoiada / area_total) * 100
        
        return penalidade
    
    def calcula_penalidade_ordem(self, posicoes):
        """Calcula penalidade para violações de ordem de entrega"""
        penalidade = 0
        n = len(posicoes)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                d_i = self.caixas[i][4]
                d_j = self.caixas[j][4]
                x_i, y_i, z_i, r_i = posicoes[i]
                x_j, y_j, z_j, r_j = posicoes[j]
                
                l_i_ef, w_i_ef, h_i_ef = self.get_dimensoes_efetivas(self.caixas[i], r_i)
                l_j_ef, w_j_ef, h_j_ef = self.get_dimensoes_efetivas(self.caixas[j], r_j)
                
                # Verifica sobreposição horizontal
                sobrepoe_xy = (x_i < x_j + l_j_ef and x_i + l_i_ef > x_j and
                              y_i < y_j + w_j_ef and y_i + w_i_ef > y_j)
                
                # Caixa com maior prioridade (sai depois) não pode estar debaixo/em cima/atrás
                if d_i > d_j and sobrepoe_xy and z_i + h_i_ef == z_j:
                    penalidade += 100 * (d_i - d_j)
                elif d_i > d_j and sobrepoe_xy and z_i == z_j + h_j_ef:
                    penalidade += 100 * (d_i - d_j)
                elif d_i > d_j and x_i > x_j and sobrepoe_xy:
                    penalidade += 10 * (d_i - d_j)
                elif d_i < d_j and x_i > x_j:
                    penalidade += 5 * (d_j - d_i)
        
        return penalidade
    
    def calcula_fitness(self, posicoes, debug=False):
        """Calcula o fitness da solução"""
        volume_total = self.L * self.W * self.H
        volume_ocupado = self.calcula_volume_ocupado(posicoes)
        U = volume_ocupado / volume_total if volume_total > 0 else 0
        
        Ps = self.calcula_penalidade_instabilidade(posicoes)
        Pd = self.calcula_penalidade_ordem(posicoes)
        
        # Verificar sobreposição com debug
        tem_sobreposicao = self.verifica_todas_sobreposicoes(posicoes, debug)
        P_over = 1000 if tem_sobreposicao else 0
        
        if debug and tem_sobreposicao:
            print(f"  Fitness: U={U:.3f}, Ps={Ps:.1f}, Pd={Pd:.1f}, P_over={P_over}")
        
        return self.alpha * U - self.beta * Ps - self.gamma * Pd - P_over
    
    def decoder(self, chaves):
        """Decodifica um vetor de chaves em uma solução"""
        solucao = []
        gene_idx = 0
        
        for i, caixa in enumerate(self.caixas):
            cid, l, w, h, d, nome = caixa
            
            x_norm = chaves[gene_idx]
            y_norm = chaves[gene_idx + 1]
            z_norm = chaves[gene_idx + 2]
            r_norm = chaves[gene_idx + 3]
            gene_idx += 4
            
            r = 0 if r_norm < 0.5 else 1
            
            l_ef, w_ef, h_ef = self.get_dimensoes_efetivas(caixa, r)
            
            max_x = self.L - l_ef
            max_y = self.W - w_ef
            max_z = self.H - h_ef
            
            if max_x > 0:
                x = int(x_norm * max_x)
            else:
                x = 0
                
            if max_y > 0:
                y = int(y_norm * max_y)
            else:
                y = 0
                
            if max_z > 0:
                z = int(z_norm * max_z)
            else:
                z = 0
            
            # Garantir limites
            x = min(max(x, 0), max_x) if max_x > 0 else 0
            y = min(max(y, 0), max_y) if max_y > 0 else 0
            z = min(max(z, 0), max_z) if max_z > 0 else 0
            
            solucao.append((x, y, z, r))
        
        return solucao
    
    def criar_individuo(self):
        """Cria um indivíduo aleatório"""
        return [random.random() for _ in range(self.num_genes)]
    
    def crossover(self, pai_elite, pai_nao_elite):
        """Cruzamento enviesado"""
        filho = []
        for i in range(self.num_genes):
            if random.random() < self.prob_crossover:
                filho.append(pai_elite[i])
            else:
                filho.append(pai_nao_elite[i])
        return filho
    
    def mutacao(self, individuo, prob=0.05):
        """Aplica mutação"""
        for i in range(len(individuo)):
            if random.random() < prob:
                individuo[i] = random.random()
        return individuo
    
    def executar(self, num_geracoes):
        """Executa o algoritmo BRKGA"""
        if self.verbose:
            print("="*50)
            print("EXECUTANDO BRKGA")
            print(f"Pop: {self.num_individuos} | Elite: {self.num_elite} | Gens: {num_geracoes}")
            print("="*50)
        
        # Inicializar
        self.populacao = [self.criar_individuo() for _ in range(self.num_individuos)]
        self.fitness = []
        
        for i, ind in enumerate(self.populacao):
            sol = self.decoder(ind)
            fit = self.calcula_fitness(sol, debug=False)
            self.fitness.append(fit)
            if fit > self.melhor_fitness:
                self.melhor_fitness = fit
                self.melhor_solucao = sol.copy()
        
        self.historico_melhores.append(self.melhor_fitness)
        
        if self.verbose:
            print(f"Melhor inicial: {self.melhor_fitness:.4f}")
        
        # Evoluir
        for geracao in range(1, num_geracoes + 1):
            # Ordenar
            indices = sorted(range(self.num_individuos), key=lambda i: self.fitness[i], reverse=True)
            elite = [self.populacao[i] for i in indices[:self.num_elite]]
            nao_elite = [self.populacao[i] for i in indices[self.num_elite:]]
            
            nova_pop = elite.copy()
            
            # Cruzamento
            while len(nova_pop) < self.num_individuos - self.num_mutantes:
                pai1 = random.choice(elite)
                pai2 = random.choice(nao_elite)
                filho = self.crossover(pai1, pai2)
                nova_pop.append(self.mutacao(filho))
            
            # Mutantes
            for _ in range(self.num_mutantes):
                nova_pop.append(self.criar_individuo())
            
            self.populacao = nova_pop
            
            # Avaliar
            self.fitness = []
            for ind in self.populacao:
                sol = self.decoder(ind)
                fit = self.calcula_fitness(sol, debug=False)
                self.fitness.append(fit)
                if fit > self.melhor_fitness:
                    self.melhor_fitness = fit
                    self.melhor_solucao = sol.copy()
            
            self.historico_melhores.append(self.melhor_fitness)
            
            if self.verbose and geracao % 50 == 0:
                print(f"Geração {geracao}: {self.melhor_fitness:.4f}")
        
        # Salvar
        volume_total = self.L * self.W * self.H
        volume_ocupado = self.calcula_volume_ocupado(self.melhor_solucao)
        
        # Verificar a melhor solução com debug
        print("\n" + "="*50)
        print("VERIFICANDO MELHOR SOLUÇÃO")
        print("="*50)
        
        # Verificar sobreposição
        tem_sobreposicao = self.verifica_todas_sobreposicoes(self.melhor_solucao, debug=True)
        if tem_sobreposicao:
            print("❌ A SOLUÇÃO TEM SOBREPOSIÇÃO!")
            # Listar todas as sobreposições
            print("\nLista completa de caixas da melhor solução:")
            for i, (pos, c) in enumerate(zip(self.melhor_solucao, self.caixas)):
                _, l, w, h, p, nome = c
                l_ef, w_ef, h_ef = self.get_dimensoes_efetivas(c, pos[3])
                print(f"  {i+1}: {nome} (P{p}) - pos={pos}, dim={l_ef}x{w_ef}x{h_ef}")
        else:
            print("✅ Solução sem sobreposição!")
        
        # Verificar estabilidade
        estavel = self.verifica_estabilidade_geral(self.melhor_solucao)
        print(f"Estabilidade: {'✅ OK' if estavel else '❌ FALHA'}")
        
        # Verificar ordem
        ordem = self.verifica_ordem_entrega(self.melhor_solucao)
        print(f"Ordem de entrega: {'✅ OK' if ordem else '❌ FALHA'}")
        
        # Calcular fitness final
        fit_final = self.calcula_fitness(self.melhor_solucao, debug=True)
        print(f"Fitness final: {fit_final:.4f}")
        
        dados = {
            'melhor_fitness': self.melhor_fitness,
            'melhor_solucao': [list(pos) for pos in self.melhor_solucao],
            'historico': self.historico_melhores,
            'volume_total': volume_total,
            'volume_ocupado': volume_ocupado,
            'utilizacao': volume_ocupado / volume_total if volume_total > 0 else 0,
            'L': self.L, 'W': self.W, 'H': self.H,
            'caixas': [{'id': c[0], 'l': c[1], 'w': c[2], 'h': c[3], 'prioridade': c[4], 'nome': c[5]} for c in self.caixas]
        }
        
        with open(f"{self.pasta_resultados}/resultados.json", "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Melhor fitness: {self.melhor_fitness:.4f}")
            print(f"📦 Volume: {volume_ocupado}/{volume_total} ({volume_ocupado/volume_total:.1%})")
            print(f"💾 Salvo em: {self.pasta_resultados}/resultados.json")
        
        return self.melhor_solucao, self.melhor_fitness, self.historico_melhores


# Função auxiliar para carregar resultados
def carregar_resultados(caminho="resultados_brkga/resultados.json"):
    """Carrega resultados do arquivo JSON"""
    import os
    if not os.path.exists(caminho):
        return None
    
    with open(caminho, 'r', encoding="utf-8") as f:
        dados = json.load(f)
    
    solucao = [tuple(pos) for pos in dados['melhor_solucao']]
    caixas = [(c['id'], c['l'], c['w'], c['h'], c['prioridade'], c['nome']) for c in dados['caixas']]
    L, W, H = dados['L'], dados['W'], dados['H']
    fitness = dados['melhor_fitness']
    
    return solucao, caixas, L, W, H, fitness


# Execução apenas se for o script principal
if __name__ == "__main__":
    # Exemplo de uso
    L, W, H = 10, 8, 6
    caixas = [
        (1, 3, 2, 2, 0, "A"),
        (2, 2, 2, 3, 0, "B"),
        (3, 4, 2, 1, 1, "C"),
        (4, 2, 2, 2, 1, "D"),
        (5, 3, 2, 2, 1, "E"),
        (6, 5, 2, 1, 2, "F"),
        (7, 4, 3, 1, 2, "G"),
    ]
    
    brkga = BRKGA(
        num_genes=len(caixas)*4,
        num_individuos=50,
        num_elite=10,
        num_mutantes=5,
        prob_crossover=0.7,
        caixas=caixas,
        L=L, W=W, H=H,
        verbose=True
    )
    
    solucao, fitness, hist = brkga.executar(50)