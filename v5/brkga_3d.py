import random
import numpy as np
import json
import os
import time
from datetime import datetime
import csv

class PontoPosicionamento:
    """Representa um ponto disponível para posicionar uma caixa"""
    def __init__(self, x, y, z, largura_livre_x, largura_livre_y):
        self.x = x
        self.y = y
        self.z = z
        self.largura_livre_x = largura_livre_x
        self.largura_livre_y = largura_livre_y
    
    def __repr__(self):
        return f"Ponto({self.x},{self.y},{self.z})"


class DecodificadorConstrutivo:
    """Decodificador que posiciona caixas sequencialmente em pontos disponiveis"""
    
    def __init__(self, caixas, L, W, H, 
                 gamma_unico=None,
                 gamma_x=None, gamma_y=None, gamma_z=None,
                 peso_x=1.0, peso_y=1.0, peso_z=1.0,
                 exigir_estabilidade_descarga=True,
                 suporte_minimo=1.0):
        """
        Args:
            gamma_unico: Se fornecido, usa um único gamma para todas violações
            gamma_x, gamma_y, gamma_z: Gammas separados por eixo
            peso_x, peso_y, peso_z: Pesos para modo híbrido (gamma_unico * peso)
            exigir_estabilidade_descarga: Se True, soluções instáveis durante descarga são inválidas
            suporte_minimo: Percentual mínimo de área apoiada (1.0 = 100%)
        """
        self.caixas = caixas
        self.L = L
        self.W = W
        self.H = H
        self.exigir_estabilidade_descarga = exigir_estabilidade_descarga
        self.suporte_minimo = suporte_minimo
        
        # Configuração de penalidade de ordem
        self.gamma_unico = gamma_unico
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.gamma_z = gamma_z
        self.peso_x = peso_x
        self.peso_y = peso_y
        self.peso_z = peso_z
        
        self.posicoes_colocadas = []
        self.ocupado = set()
        self.pontos_disponiveis = []
        
        # Métricas
        self.tempo_construcao = 0
        self.tempo_fitness = 0
        self.debug = False
    
    def inicializar_pontos(self):
        self.pontos_disponiveis = [PontoPosicionamento(0, 0, 0, self.L, self.W)]
    
    def ponto_valido(self, ponto, l_ef, w_ef, h_ef, nova_prioridade):
        # 1. Verificar limites do container
        if ponto.x + l_ef > self.L or ponto.y + w_ef > self.W or ponto.z + h_ef > self.H:
            return False
        
        # 2. Verificar colisões
        for dx in range(l_ef):
            for dy in range(w_ef):
                for dz in range(h_ef):
                    if (ponto.x + dx, ponto.y + dy, ponto.z + dz) in self.ocupado:
                        return False
        
        # 3. RESTRIÇÃO RÍGIDA EM X (NOVO!)
        # Verificar se a nova caixa não cria violação de ordem no eixo X
        for caixa in self.posicoes_colocadas:
            x_existente = caixa['posicao'][0]
            y_existente = caixa['posicao'][1]
            z_existente = caixa['posicao'][2]
            prioridade_existente = caixa['prioridade']
            l_existente, w_existente, h_existente = caixa['dimensoes']
            
            # Verificar sobreposição em Y e Z (mesma "coluna")
            y_overlap = (ponto.y < y_existente + w_existente) and (ponto.y + w_ef > y_existente)
            z_overlap = (ponto.z < z_existente + h_existente) and (ponto.z + h_ef > z_existente)
            
            if y_overlap and z_overlap:
                # Mesma coluna em Y/Z - verificar ordem em X
                if nova_prioridade < prioridade_existente:
                    # Nova caixa é MAIS prioritária
                    if ponto.x > x_existente:
                        # Está ATRÁS da caixa existente → PROIBIDO!
                        if self.debug:
                            print(f"  [RESTRIÇÃO X] Caixa prioritária {nova_prioridade} não pode ficar atrás de {prioridade_existente}")
                        return False
                elif nova_prioridade > prioridade_existente:
                    # Nova caixa é MENOS prioritária
                    if ponto.x < x_existente:
                        # Está NA FRENTE da caixa prioritária → PROIBIDO!
                        if self.debug:
                            print(f"  [RESTRIÇÃO X] Caixa {nova_prioridade} não pode ficar na frente de {prioridade_existente}")
                        return False
        
        return True

    
    def colocar_caixa(self, ponto, caixa, rotacao, indice_caixa):
        _, l, w, h, prioridade, nome = caixa
        l_ef = l if rotacao == 0 else w
        w_ef = w if rotacao == 0 else l
        h_ef = h
        
        self.posicoes_colocadas.append({
            'indice': indice_caixa,
            'nome': nome,
            'prioridade': prioridade,
            'posicao': (ponto.x, ponto.y, ponto.z, rotacao),
            'dimensoes': (l_ef, w_ef, h_ef)
        })
        
        for dx in range(l_ef):
            for dy in range(w_ef):
                for dz in range(h_ef):
                    self.ocupado.add((ponto.x + dx, ponto.y + dy, ponto.z + dz))
        
        self.atualizar_pontos(ponto, l_ef, w_ef, h_ef)
    
    def atualizar_pontos(self, ponto, l_ef, w_ef, h_ef):
        self.pontos_disponiveis.remove(ponto)
        
        ponto_topo = PontoPosicionamento(ponto.x, ponto.y, ponto.z + h_ef, l_ef, w_ef)
        self.pontos_disponiveis.append(ponto_topo)
        
        if ponto.x + l_ef < self.L:
            largura_x = self.L - (ponto.x + l_ef)
            ponto_direita = PontoPosicionamento(ponto.x + l_ef, ponto.y, ponto.z, largura_x, w_ef)
            self.pontos_disponiveis.append(ponto_direita)
        
        if ponto.y + w_ef < self.W:
            largura_y = self.W - (ponto.y + w_ef)
            ponto_frente = PontoPosicionamento(ponto.x, ponto.y + w_ef, ponto.z, l_ef, largura_y)
            self.pontos_disponiveis.append(ponto_frente)
        
        self.pontos_disponiveis.sort(key=lambda p: (p.z, p.x, p.y))
    
    def ponto_tem_suporte(self, ponto, l_ef, w_ef):
        """Verifica se o ponto tem suporte total (100% da área apoiada)"""
        if ponto.z == 0:
            return True
        
        area_total = l_ef * w_ef
        area_apoiada = 0
        
        for dx in range(l_ef):
            for dy in range(w_ef):
                if (ponto.x + dx, ponto.y + dy, ponto.z - 1) in self.ocupado:
                    area_apoiada += 1
        
        return area_apoiada >= area_total * self.suporte_minimo
    
    def encontrar_melhor_ponto(self, l_ef, w_ef, h_ef,nova_prioridade):
        """Bottom-Left-Front: prioriza menor z, depois menor x, depois menor y"""
        melhores_pontos = []
        
        for ponto in self.pontos_disponiveis:
            if self.ponto_valido(ponto, l_ef, w_ef, h_ef,nova_prioridade):
                tem_suporte = self.ponto_tem_suporte(ponto, l_ef, w_ef)
                
                # BLF: menor z, depois menor x, depois menor y
                # Pontos sem suporte são descartados (não podem ser usados)
                if not tem_suporte:
                    continue
                
                # Peso baseado na altura (priorizar pontos mais baixos)
                peso = ponto.z  # Quanto menor, melhor
                
                melhores_pontos.append((peso, ponto.x, ponto.y, ponto))
        
        if not melhores_pontos:
            return None
        
        # Ordenar por (z, x, y) - Bottom-Left-Front
        melhores_pontos.sort(key=lambda x: (x[0], x[1], x[2]))
        return melhores_pontos[0][3]
    
    def construir_solucao(self, ordem_caixas, rotacoes):
        inicio_const = time.time()
        
        self.posicoes_colocadas = []
        self.ocupado = set()
        self.inicializar_pontos()
        
        for i_caixa in ordem_caixas:
            caixa = self.caixas[i_caixa]
            _, l, w, h, prioridade, nome = caixa
            rotacao = rotacoes[i_caixa]
            
            l_ef = l if rotacao == 0 else w
            w_ef = w if rotacao == 0 else l
            h_ef = h
            
            ponto = self.encontrar_melhor_ponto(l_ef, w_ef, h_ef,prioridade)
            
            if ponto is None:
                self.tempo_construcao += time.time() - inicio_const
                return None
            
            self.colocar_caixa(ponto, caixa, rotacao, i_caixa)
        
        self.tempo_construcao += time.time() - inicio_const
        return self.posicoes_colocadas
    
    def verificar_estabilidade_configuracao(self, posicoes, ocupado):
        """Verifica se todas as caixas em uma configuração estão estáveis (100% apoio)"""
        for p in posicoes:
            x, y, z, r = p['posicao']
            l_ef, w_ef, h_ef = p['dimensoes']
            
            if z == 0:
                continue
            
            area_total = l_ef * w_ef
            area_apoiada = 0
            
            for dx in range(l_ef):
                for dy in range(w_ef):
                    if (x + dx, y + dy, z - 1) in ocupado:
                        area_apoiada += 1
            
            if area_apoiada < area_total * self.suporte_minimo:
                return False
        return True
    
    def verificar_descarga_segura(self):
        """Verifica se a descarga completa é segura (100% apoio em todos os estágios)"""
        if not self.posicoes_colocadas:
            return True, None, []
        
        # Agrupar caixas por prioridade
        posicoes_por_prioridade = {}
        for p in self.posicoes_colocadas:
            prioridade = p['prioridade']
            if prioridade not in posicoes_por_prioridade:
                posicoes_por_prioridade[prioridade] = []
            posicoes_por_prioridade[prioridade].append(p)
        
        prioridades_ordenadas = sorted(posicoes_por_prioridade.keys())
        
        # Simular descarga progressiva
        ocupado_atual = self.ocupado.copy()
        caixas_restantes = self.posicoes_colocadas.copy()
        
        for i, p_atual in enumerate(prioridades_ordenadas):
            # Remover caixas da prioridade atual
            caixas_remover = posicoes_por_prioridade[p_atual]
            
            for caixa in caixas_remover:
                x, y, z, r = caixa['posicao']
                l_ef, w_ef, h_ef = caixa['dimensoes']
                
                for dx in range(l_ef):
                    for dy in range(w_ef):
                        for dz in range(h_ef):
                            ocupado_atual.discard((x + dx, y + dy, z + dz))
                
                caixas_restantes.remove(caixa)
            
            # Verificar estabilidade das caixas restantes
            if len(caixas_restantes) > 0:
                caixas_instaveis = []
                for p in caixas_restantes:
                    x, y, z, r = p['posicao']
                    l_ef, w_ef, h_ef = p['dimensoes']
                    
                    if z == 0:
                        continue
                    
                    area_total = l_ef * w_ef
                    area_apoiada = 0
                    
                    for dx in range(l_ef):
                        for dy in range(w_ef):
                            if (x + dx, y + dy, z - 1) in ocupado_atual:
                                area_apoiada += 1
                    
                    if area_apoiada < area_total * self.suporte_minimo:
                        caixas_instaveis.append({
                            'nome': p['nome'],
                            'prioridade': p['prioridade'],
                            'posicao': (x, y, z),
                            'area_apoiada': area_apoiada,
                            'area_total': area_total
                        })
                
                if caixas_instaveis:
                    return False, p_atual, caixas_instaveis
        
        return True, None, []
    
    def calcular_penalidade_ordem(self, posicoes_colocadas):
        """
        Calcula penalidade para ordem de retirada
        
        Suporta três modos:
        1. gamma_unico: penalidade = gamma_unico * (viol_x + viol_y + viol_z)
        2. gamma separado: penalidade = gamma_x*viol_x + gamma_y*viol_y + gamma_z*viol_z
        3. Híbrido: penalidade = gamma_unico * (peso_x*viol_x + peso_y*viol_y + peso_z*viol_z)
        """
        if len(posicoes_colocadas) < 2:
            return 0
        
        violacao_x = 0
        violacao_y = 0
        violacao_z = 0
        total_pares = 0
        
        for i, p1 in enumerate(posicoes_colocadas):
            for j, p2 in enumerate(posicoes_colocadas):
                if i == j:
                    continue
                
                # Se p1 tem maior prioridade (menor número) que p2
                if p1['prioridade'] < p2['prioridade']:
                    total_pares += 1
                    diff_prioridade = p2['prioridade'] - p1['prioridade']
                    peso_prioridade = diff_prioridade / 100.0
                    
                    # Eixo X: p1 atrás de p2
                    if p1['posicao'][0] > p2['posicao'][0]:
                        violacao_x += peso_prioridade
                    
                    # Eixo Y: desalinhamento lateral
                    if abs(p1['posicao'][1] - p2['posicao'][1]) > 0:
                        violacao_y += peso_prioridade * 0.5
                    
                    # Eixo Z: p1 embaixo de p2
                    if p1['posicao'][2] < p2['posicao'][2]:
                        violacao_z += peso_prioridade
        
        # Normalizar
        if total_pares > 0:
            violacao_x = violacao_x / total_pares
            violacao_y = violacao_y / total_pares
            violacao_z = violacao_z / total_pares
        
        # Aplicar modo de penalidade
        if self.gamma_unico is not None:
            # Modo 1: gamma único OU modo híbrido (com pesos)
            if self.peso_x != 1.0 or self.peso_y != 1.0 or self.peso_z != 1.0:
                # Modo híbrido: gamma_unico * (pesos * violações)
                penalidade = self.gamma_unico * (
                    self.peso_x * violacao_x +
                    self.peso_y * violacao_y +
                    self.peso_z * violacao_z
                )
            else:
                # Modo gamma único simples
                penalidade = self.gamma_unico * (violacao_x + violacao_y + violacao_z)
        else:
            # Modo gammas separados
            # Modo gammas separados (X pode ser None se for restrição rígida)
            gx = self.gamma_x if self.gamma_x is not None else 0.0
            gy = self.gamma_y if self.gamma_y is not None else 0.0
            gz = self.gamma_z if self.gamma_z is not None else 0.0
            
            penalidade = (gx * violacao_x +
                        gy * violacao_y +
                        gz * violacao_z)
            
        return penalidade
    
    def calcular_fitness(self, posicoes_colocadas):
        """Calcula fitness com restrição rígida para descarga segura"""
        inicio_fit = time.time()
        
        if posicoes_colocadas is None:
            return -float('inf')
        
        # RESTRIÇÃO RÍGIDA: Verificar segurança da descarga
        if self.exigir_estabilidade_descarga:
            is_safe, prioridade_violacao, caixas_instaveis = self.verificar_descarga_segura()
            
            if not is_safe:
                if self.debug:
                    print(f"  [RESTRIÇÃO] Descarga insegura na prioridade {prioridade_violacao}")
                self.tempo_fitness += time.time() - inicio_fit
                return -float('inf')
        
        # Penalidade de ordem (único componente do fitness)
        penalidade_ordem = self.calcular_penalidade_ordem(posicoes_colocadas)
        
        # Fitness é o negativo da penalidade (menor penalidade = melhor fitness)
        fitness = -penalidade_ordem
        
        self.tempo_fitness += time.time() - inicio_fit
        return fitness
    
    def obter_solucao_formatada(self, posicoes_colocadas):
        if posicoes_colocadas is None:
            return None
        
        solucao = [None] * len(self.caixas)
        for p in posicoes_colocadas:
            solucao[p['indice']] = p['posicao']
        
        return solucao if None not in solucao else None
    
    def validar_descarga_completa(self):
        """Valida se a descarga é completamente segura"""
        is_safe, prioridade, instaveis = self.verificar_descarga_segura()
        
        if is_safe:
            print("✅ DESCARGA SEGURA: Nenhuma caixa ficará voando durante a descarga")
        else:
            print(f"❌ DESCARGA INSEGURA: Problema ao remover prioridade {prioridade}")
            print(f"   Caixas que ficariam instáveis:")
            for c in instaveis:
                print(f"     - {c['nome']} (prioridade {c['prioridade']})")
        
        return is_safe


class BRKGA:
    """Implementação do BRKGA para otimização da ordem de posicionamento"""
    
    def __init__(self, num_genes, num_individuos, num_elite, num_mutantes, prob_crossover, 
                 caixas, L, W, H, 
                 gamma_unico=None,
                 gamma_x=None, gamma_y=None, gamma_z=None,
                 peso_x=1.0, peso_y=1.0, peso_z=1.0,
                 exigir_estabilidade_descarga=True,
                 suporte_minimo=1.0,
                 verbose=True):
        
        self.num_genes = num_genes
        self.num_individuos = num_individuos
        self.num_elite = num_elite
        self.num_mutantes = num_mutantes
        self.prob_crossover = prob_crossover
        self.caixas = caixas
        self.L = L
        self.W = W
        self.H = H
        self.exigir_estabilidade_descarga = exigir_estabilidade_descarga
        self.suporte_minimo = suporte_minimo
        self.verbose = verbose
        
        # Parâmetros de penalidade
        self.gamma_unico = gamma_unico
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.gamma_z = gamma_z
        self.peso_x = peso_x
        self.peso_y = peso_y
        self.peso_z = peso_z
        
        self.num_caixas = len(caixas)
        
        self.populacao = []
        self.fitness = []
        self.melhor_solucao = None
        self.melhor_fitness = -float('inf')
        self.melhor_cromossomo = None
        self.historico_melhores = []
        
        self.tempos = {
            'inicio': None,
            'fim': None,
            'total_segundos': 0,
            'tempo_por_geracao': [],
            'tempo_avaliacao_por_geracao': [],
            'tempo_crossover_por_geracao': [],
            'tempo_decodificacao_total': 0,
            'tempo_fitness_total': 0
        }
        
        self.registro_geracoes = []
        self.geracao_atual = 0
        self.motivo_parada = None
        self.solucoes_invalidas = 0
        
        self.pasta_resultados = "resultados_brkga"
        os.makedirs(self.pasta_resultados, exist_ok=True)
    
    def decodificar_individuo(self, chaves):
        chaves_ordem = chaves[:self.num_caixas]
        ordem = sorted(range(self.num_caixas), key=lambda i: chaves_ordem[i])
        chaves_rotacao = chaves[self.num_caixas:]
        rotacoes = [0 if r < 0.5 else 1 for r in chaves_rotacao]
        return ordem, rotacoes
    
    def salvar_resultados_completos(self):
        """Salva todos os resultados em uma pasta com timestamp"""
        
        # Criar pasta com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pasta_execucao = os.path.join(self.pasta_resultados, f"execucao_{timestamp}")
        os.makedirs(pasta_execucao, exist_ok=True)
        
        # Preparar dados completos da solução
        dados_completos = {
            'metadata': {
                'timestamp': timestamp,
                'data_execucao': datetime.now().isoformat(),
                'motivo_parada': self.motivo_parada,
                'geracoes_executadas': self.geracao_atual,
                'melhor_fitness': self.melhor_fitness,
                'volume_total': self.L * self.W * self.H,
                'volume_ocupado': self.volume_ocupado if hasattr(self, 'volume_ocupado') else 0,
                'utilizacao': self.volume_ocupado / (self.L * self.W * self.H) if hasattr(self, 'volume_ocupado') else 0,
                'solucoes_invalidas': self.solucoes_invalidas
            },
            'parametros': {
                'num_individuos': self.num_individuos,
                'num_elite': self.num_elite,
                'num_mutantes': self.num_mutantes,
                'prob_crossover': self.prob_crossover,
                'num_caixas': self.num_caixas,
                'L': self.L, 'W': self.W, 'H': self.H,
                'gamma_unico': self.gamma_unico,
                'gamma_x': self.gamma_x, 'gamma_y': self.gamma_y, 'gamma_z': self.gamma_z,
                'peso_x': self.peso_x, 'peso_y': self.peso_y, 'peso_z': self.peso_z,
                'exigir_estabilidade_descarga': self.exigir_estabilidade_descarga,
                'suporte_minimo': self.suporte_minimo
            },
            'caixas': [
                {
                    'id': c[0], 
                    'l': c[1], 
                    'w': c[2], 
                    'h': c[3], 
                    'prioridade': c[4], 
                    'nome': c[5]
                } 
                for c in self.caixas
            ],
            'melhor_solucao': self.melhor_solucao,
            'historico_fitness': self.historico_melhores,
            'tempos': self.tempos
        }
        
        # Salvar JSON completo
        with open(os.path.join(pasta_execucao, 'solucao_completa.json'), 'w', encoding='utf-8') as f:
            json.dump(dados_completos, f, indent=2)
        
        # Salvar CSV com posições das caixas
        with open(os.path.join(pasta_execucao, 'posicoes_caixas.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['indice', 'nome', 'prioridade', 'x', 'y', 'z', 'rotacao', 'l', 'w', 'h'])
            
            if self.melhor_solucao:
                for i, pos in enumerate(self.melhor_solucao):
                    if pos:
                        x, y, z, r = pos
                        _, l, w, h, prioridade, nome = self.caixas[i]
                        l_ef = l if r == 0 else w
                        w_ef = w if r == 0 else l
                        writer.writerow([i, nome, prioridade, x, y, z, r, l_ef, w_ef, h])
        
        # Salvar registro de gerações
        with open(os.path.join(pasta_execucao, 'registro_geracoes.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['geracao', 'melhor_fitness', 'tempo_acumulado', 'melhoria'])
            for reg in self.registro_geracoes:
                writer.writerow([
                    reg['geracao'], reg['melhor_fitness'], 
                    reg['tempo_acumulado'], reg['melhoria']
                ])
        
        print(f"📁 Resultados salvos em: {pasta_execucao}")
        
        return pasta_execucao

    def avaliar_individuo(self, chaves):
        ordem, rotacoes = self.decodificar_individuo(chaves)
        
        inicio_dec = time.time()
        decodificador = DecodificadorConstrutivo(
            self.caixas, self.L, self.W, self.H,
            gamma_unico=self.gamma_unico,
            gamma_x=self.gamma_x, gamma_y=self.gamma_y, gamma_z=self.gamma_z,
            peso_x=self.peso_x, peso_y=self.peso_y, peso_z=self.peso_z,
            exigir_estabilidade_descarga=self.exigir_estabilidade_descarga,
            suporte_minimo=self.suporte_minimo
        )
        posicoes = decodificador.construir_solucao(ordem, rotacoes)
        self.tempos['tempo_decodificacao_total'] += time.time() - inicio_dec
        
        if posicoes is None:
            self.solucoes_invalidas += 1
            return -float('inf'), None
        
        fitness = decodificador.calcular_fitness(posicoes)
        
        if fitness == -float('inf'):
            self.solucoes_invalidas += 1
        
        self.tempos['tempo_fitness_total'] += decodificador.tempo_fitness
        solucao = decodificador.obter_solucao_formatada(posicoes)
        
        return fitness, solucao
    
    def criar_individuo(self):
        return [random.random() for _ in range(self.num_genes)]
    
    def crossover(self, pai_elite, pai_nao_elite):
        filho = []
        for i in range(self.num_genes):
            if random.random() < self.prob_crossover:
                filho.append(pai_elite[i])
            else:
                filho.append(pai_nao_elite[i])
        return filho
    
    def salvar_checkpoint(self):
        checkpoint = {
            'geracao': self.geracao_atual,
            'melhor_fitness': self.melhor_fitness,
            'melhor_solucao': self.melhor_solucao,
            'historico': self.historico_melhores,
            'solucoes_invalidas': self.solucoes_invalidas,
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{self.pasta_resultados}/checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
    
    def salvar_metricas(self):
        with open(f"{self.pasta_resultados}/registro_geracoes.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['geracao', 'melhor_fitness', 'tempo_acumulado', 
                           'tempo_geracao', 'melhoria', 'fitness_melhorou'])
            for reg in self.registro_geracoes:
                writer.writerow([
                    reg['geracao'], reg['melhor_fitness'], 
                    reg['tempo_acumulado'], reg['tempo_geracao'],
                    reg['melhoria'], reg['fitness_melhorou']
                ])
        
        resumo = {
            'data_execucao': datetime.now().isoformat(),
            'motivo_parada': self.motivo_parada,
            'parametros': {
                'num_individuos': self.num_individuos,
                'num_elite': self.num_elite,
                'num_mutantes': self.num_mutantes,
                'prob_crossover': self.prob_crossover,
                'num_caixas': self.num_caixas,
                'L': self.L, 'W': self.W, 'H': self.H,
                'gamma_unico': self.gamma_unico,
                'gamma_x': self.gamma_x, 'gamma_y': self.gamma_y, 'gamma_z': self.gamma_z,
                'peso_x': self.peso_x, 'peso_y': self.peso_y, 'peso_z': self.peso_z,
                'exigir_estabilidade_descarga': self.exigir_estabilidade_descarga,
                'suporte_minimo': self.suporte_minimo
            },
            'resultados': {
                'melhor_fitness': self.melhor_fitness,
                'volume_ocupado': self.volume_ocupado if hasattr(self, 'volume_ocupado') else 0,
                'volume_total': self.L * self.W * self.H,
                'geracoes_executadas': self.geracao_atual,
                'solucoes_invalidas': self.solucoes_invalidas
            }
        }
        
        with open(f"{self.pasta_resultados}/metricas_tempo.json", "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2)
        
        print(f"Metricas salvas em: {self.pasta_resultados}/")
    
    def executar_continuo(self, tempo_maximo_horas=None, geracoes_max=None, 
                          sem_melhoria_max=1000, fitness_alvo=None):
        
        inicio = time.time()
        self.tempos['inicio'] = inicio
        tempo_max_segundos = tempo_maximo_horas * 3600 if tempo_maximo_horas else None
        
        if self.verbose:
            print("="*50)
            print("BRKGA - OTIMIZACAO COM RESTRICAO DE DESCARGA SEGURA")
            print(f"Pop: {self.num_individuos} | Elite: {self.num_elite}")
            print(f"Caixas: {self.num_caixas} | Genes: {self.num_genes}")
            if self.gamma_unico is not None:
                if self.peso_x != 1.0 or self.peso_y != 1.0 or self.peso_z != 1.0:
                    print(f"Modo HÍBRIDO: γ={self.gamma_unico}, px={self.peso_x}, py={self.peso_y}, pz={self.peso_z}")
                else:
                    print(f"Modo ÚNICO: γ={self.gamma_unico}")
            else:
                print(f"Modo SEPARADO: γx={self.gamma_x}, γy={self.gamma_y}, γz={self.gamma_z}")
            print(f"Descarga segura obrigatória: {self.exigir_estabilidade_descarga}")
            print(f"Suporte mínimo: {self.suporte_minimo*100:.0f}%")
            print("="*50)
        
        # Inicializar populacao
        self.populacao = [self.criar_individuo() for _ in range(self.num_individuos)]
        self.fitness = []
        
        for ind in self.populacao:
            fit, sol = self.avaliar_individuo(ind)
            self.fitness.append(fit)
            if fit > self.melhor_fitness:
                self.melhor_fitness = fit
                self.melhor_solucao = sol
                self.melhor_cromossomo = ind.copy()
        
        self.historico_melhores.append(self.melhor_fitness)
        
        if self.verbose:
            print(f"Melhor inicial: {self.melhor_fitness}")
            print(f"Soluções inválidas: {self.solucoes_invalidas}")
        
        geracao = 0
        ultima_melhoria = geracao
        tempo_ultimo_print = inicio
        
        while True:
            inicio_geracao = time.time()
            geracao += 1
            self.geracao_atual = geracao
            
            # Crossover
            inicio_cross = time.time()
            indices = sorted(range(self.num_individuos), key=lambda i: self.fitness[i], reverse=True)
            elite = [self.populacao[i] for i in indices[:self.num_elite]]
            nao_elite = [self.populacao[i] for i in indices[self.num_elite:]]
            
            nova_pop = elite.copy()
            while len(nova_pop) < self.num_individuos - self.num_mutantes:
                pai1 = random.choice(elite)
                pai2 = random.choice(nao_elite)
                filho = self.crossover(pai1, pai2)
                nova_pop.append(filho)
            
            for _ in range(self.num_mutantes):
                nova_pop.append(self.criar_individuo())
            
            tempo_cross = time.time() - inicio_cross
            self.tempos['tempo_crossover_por_geracao'].append(tempo_cross)
            self.populacao = nova_pop
            
            # Avaliacao
            inicio_avaliacao = time.time()
            self.fitness = []
            houve_melhoria = False
            
            for ind in self.populacao:
                fit, sol = self.avaliar_individuo(ind)
                self.fitness.append(fit)
                if fit > self.melhor_fitness:
                    self.melhor_fitness = fit
                    self.melhor_solucao = sol
                    self.melhor_cromossomo = ind.copy()
                    houve_melhoria = True
                    ultima_melhoria = geracao
                    self.salvar_checkpoint()
                    
                    if self.verbose:
                        print(f"[!! MELHORIA !!] Geracao {geracao}: Fitness = {self.melhor_fitness:.12f}")
            
            tempo_avaliacao = time.time() - inicio_avaliacao
            self.tempos['tempo_avaliacao_por_geracao'].append(tempo_avaliacao)
            self.historico_melhores.append(self.melhor_fitness)
            
            tempo_geracao = time.time() - inicio_geracao
            self.tempos['tempo_por_geracao'].append(tempo_geracao)
            
            self.registro_geracoes.append({
                'geracao': geracao,
                'melhor_fitness': self.melhor_fitness,
                'tempo_acumulado': time.time() - inicio,
                'tempo_geracao': tempo_geracao,
                'melhoria': geracao - ultima_melhoria,
                'fitness_melhorou': houve_melhoria
            })
            
            # Progresso
            if self.verbose and (geracao % 50 == 0 or (time.time() - tempo_ultimo_print) > 30):
                print(f"[...progresso] Geracao {geracao}: Fitness = {self.melhor_fitness:.12f}")
                print(f"   Soluções inválidas: {self.solucoes_invalidas}")
                print(f"   Sem melhoria ha {geracao - ultima_melhoria} geracoes")
                tempo_ultimo_print = time.time()
            
            # Criterios de parada
            if fitness_alvo and self.melhor_fitness >= fitness_alvo:
                self.motivo_parada = f"FITNESS_ALVO - Valor {fitness_alvo} atingido"
                break
            
            if geracoes_max and geracao >= geracoes_max:
                self.motivo_parada = f"GERACOES_MAX - Limite de {geracoes_max}"
                break
            
            if tempo_max_segundos and (time.time() - inicio) > tempo_max_segundos:
                self.motivo_parada = f"TEMPO_MAX - Limite de {tempo_maximo_horas}h"
                break
            
            if sem_melhoria_max and (geracao - ultima_melhoria) >= sem_melhoria_max:
                self.motivo_parada = f"ESTAGNACAO - {sem_melhoria_max} geracoes sem melhoria"
                break
        
        self.tempos['fim'] = time.time()
        self.tempos['total_segundos'] = self.tempos['fim'] - inicio
        
        # Calcular volume ocupado (apenas para informação)
        self.volume_ocupado = 0
        if self.melhor_solucao:
            for i, pos in enumerate(self.melhor_solucao):
                if pos:
                    _, l, w, h, _, _ = self.caixas[i]
                    x, y, z, r = pos
                    l_ef = l if r == 0 else w
                    w_ef = w if r == 0 else l
                    self.volume_ocupado += l_ef * w_ef * h
        
        self.salvar_metricas()

        # No final do método executar_continuo, após calcular volume_ocupado
        pasta_resultados = self.salvar_resultados_completos()
        
        # Validar solução final
        if self.melhor_cromossomo and self.verbose:
            print("\n" + "="*50)
            print("VALIDANDO SOLUCAO FINAL")
            print("="*50)
            
            dec_validacao = DecodificadorConstrutivo(
                self.caixas, self.L, self.W, self.H,
                gamma_unico=self.gamma_unico,
                gamma_x=self.gamma_x, gamma_y=self.gamma_y, gamma_z=self.gamma_z,
                peso_x=self.peso_x, peso_y=self.peso_y, peso_z=self.peso_z,
                exigir_estabilidade_descarga=True,
                suporte_minimo=self.suporte_minimo
            )
            
            chaves_ordem = self.melhor_cromossomo[:self.num_caixas]
            ordem = sorted(range(self.num_caixas), key=lambda i: chaves_ordem[i])
            chaves_rotacao = self.melhor_cromossomo[self.num_caixas:]
            rotacoes = [0 if r < 0.5 else 1 for r in chaves_rotacao]
            
            posicoes = dec_validacao.construir_solucao(ordem, rotacoes)
            
            if posicoes:
                dec_validacao.validar_descarga_completa()
            else:
                print("❌ Não foi possível reconstruir a solução para validação")
        
        if self.verbose:
            print("\n" + "="*50)
            print("RESULTADO FINAL")
            print("="*50)
            print(f"Motivo: {self.motivo_parada}")
            print(f"Geracoes: {geracao}")
            print(f"Fitness: {self.melhor_fitness:.12f}")
            print(f"Volume: {self.volume_ocupado}/{self.L*self.W*self.H} ({self.volume_ocupado/(self.L*self.W*self.H):.1%})")
            print(f"Soluções inválidas: {self.solucoes_invalidas}")
        
        return self.melhor_solucao, self.melhor_fitness, self.historico_melhores, self.motivo_parada


def carregar_resultados(caminho="resultados_brkga/resultados.json"):
    if not os.path.exists(caminho):
        return None
    
    with open(caminho, 'r', encoding="utf-8") as f:
        dados = json.load(f)
    
    solucao = dados['melhor_solucao']
    caixas = [(c['id'], c['l'], c['w'], c['h'], c['prioridade'], c['nome']) for c in dados['caixas']]
    L, W, H = dados['L'], dados['W'], dados['H']
    fitness = dados['melhor_fitness']
    
    return solucao, caixas, L, W, H, fitness


def converter_para_caixas(dados):
    """
    Converte o formato de dicionário para lista de caixas no formato do BRKGA.
    
    Formato de entrada:
    {
        id: {(l, w, h): quantidade, (l2, w2, h2): quantidade2, ...},
        ...
    }
    
    Formato de saida:
    [
        (id, l, w, h, prioridade, nome),
        (id, l, w, h, prioridade, nome),
        ...
    ]
    """
    caixas = []
    idx_global = 0
    
    for id_grupo, tipos in dados.items():
        for (l, w, h), quantidade in tipos.items():
            for i in range(quantidade):
                # Prioridade baseada no id do grupo (menor id = maior prioridade)
                prioridade = id_grupo + 1  # Para prioridades comecarem em 1
                nome = f"Box_{id_grupo}_{l}x{w}x{h}_{i+1}"
                caixas.append((idx_global, l, w, h, prioridade, nome))
                idx_global += 1
    
    return caixas

# Exemplo de uso


def main():
    # Suas caixas

    pedidos_order = {
        0: {(1, 1, 1): 4},
        1: {(1, 1, 4): 2},
        2: {(4, 1, 1): 2},
        3: {(3, 3, 3): 1},
        4: {(1, 1, 1): 10},
        5: {(2, 2, 1): 3, (1, 2, 2): 1}
    }
    
    caixas = converter_para_caixas(pedidos_order)

    L, W, H = 4, 5, 6
    
    # Exemplo 1: Modo ÚNICO (gamma único)
    print("\n" + "="*60)
    # print("TESTE MODO ÚNICO (γ=1.0)")
    print("TESTE com gamma_x=None, gamma_y=0 e gamma_z=0.01")
    print("="*60)
    
    brkga_unico = BRKGA(
        num_genes=len(caixas)*2,
        num_individuos=50,
        num_elite=10,
        num_mutantes=5,
        prob_crossover=0.7,
        caixas=caixas,
        L=L, W=W, H=H,
        # gamma_unico=0.05, # ficou pior no geral, gamma=0 fica melhor.
        gamma_x=0.0, gamma_y=0.00, gamma_z=0.01, #decidir****
        peso_x=1.0, peso_y=1.0, peso_z=1.0,
        exigir_estabilidade_descarga=True,
        suporte_minimo=1.0,
        verbose=True
    )
    
    solucao, fitness, historico, motivo = brkga_unico.executar_continuo(
        geracoes_max=3000,
        sem_melhoria_max=1500,
        tempo_maximo_horas=10/60,
        fitness_alvo=-0.00000001
    )
    


if __name__ == "__main__":
    print("brkga_3d.py rodado diretamente, rodando main.")
    main()