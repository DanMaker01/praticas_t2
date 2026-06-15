"""
Testes de perturbação ao redor do γ ótimo (0.05)
Objetivo: Verificar se 0.05 é realmente o ponto ótimo ou se há valores melhores próximos
"""

import json
import os
import time
import random
import numpy as np
from datetime import datetime
import csv
from brkga_3d import BRKGA

class TestePerturbacao:
    """Testa pequenas variações ao redor do γ ótimo"""
    
    def __init__(self, L, W, H, num_execucoes=5, geracoes_por_teste=100):
        self.L = L
        self.W = W
        self.H = H
        self.volume_container = L * W * H
        self.num_execucoes = num_execucoes
        self.geracoes_por_teste = geracoes_por_teste
        
        self.pasta_testes = f"testes_perturbacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
    
    def criar_caixas_cubicas(self, n, tamanho=1):
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, tamanho, tamanho, tamanho, prioridade, f"Cubo_{i}"))
        return caixas
    
    def criar_caixas_1x1x2(self, n):
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 1, 2, prioridade, f"Alta_{i}"))
        return caixas
    
    def criar_caixas_1x2x1(self, n):
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 2, 1, prioridade, f"LargaY_{i}"))
        return caixas
    
    def criar_caixas_2x1x1(self, n):
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 2, 1, 1, prioridade, f"LongaX_{i}"))
        return caixas
    
    def criar_caixas_aleatorias(self, n, min_dim=1, max_dim=3):
        caixas = []
        for i in range(n):
            l = random.randint(min_dim, max_dim)
            w = random.randint(min_dim, max_dim)
            h = random.randint(min_dim, max_dim)
            prioridade = (i % 5) + 1
            caixas.append((i, l, w, h, prioridade, f"Rand_{i}"))
        return caixas
    
    def testar_gamma(self, caixas, gamma, nome_cenario, descricao):
        """Testa um valor específico de gamma"""
        
        print(f"    γ={gamma:.4f}...", end=" ", flush=True)
        
        fitnesses = []
        tempos = []
        volumes = []
        
        for execucao in range(self.num_execucoes):
            brkga = BRKGA(
                num_genes=len(caixas)*2,
                num_individuos=50,
                num_elite=10,
                num_mutantes=5,
                prob_crossover=0.7,
                caixas=caixas,
                L=self.L, W=self.W, H=self.H,
                gamma_unico=gamma,
                exigir_estabilidade_descarga=True,
                suporte_minimo=1.0,
                verbose=False
            )
            
            inicio = time.time()
            solucao, fitness, historico, motivo = brkga.executar_continuo(
                geracoes_max=self.geracoes_por_teste,
                sem_melhoria_max=20
            )
            tempo = time.time() - inicio
            
            if fitness != -float('inf'):
                fitnesses.append(fitness)
                tempos.append(tempo)
                volumes.append(brkga.volume_ocupado)
        
        if not fitnesses:
            print(f"❌ Sem soluções")
            return None
        
        print(f"✅ {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}")
        
        return {
            'gamma': gamma,
            'fitness_mean': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'fitness_min': min(fitnesses),
            'fitness_max': max(fitnesses),
            'tempo_mean': np.mean(tempos),
            'volume_mean': np.mean(volumes),
            'sucesso': len(fitnesses) / self.num_execucoes
        }
    
    def executar_para_cenario(self, caixas, nome, descricao, gammas):
        """Executa testes para um cenário com múltiplos gammas"""
        
        print(f"\n{'='*70}")
        print(f"CENÁRIO: {nome}")
        print(f"Descrição: {descricao}")
        print(f"Caixas: {len(caixas)}")
        print(f"Volume: {sum(c[1]*c[2]*c[3] for c in caixas)}/{self.volume_container}")
        print(f"OBS: Violação em X é PROIBIDA (restrição rígida)")
        print(f"{'='*70}")
        
        resultados = []
        for gamma in gammas:
            resultado = self.testar_gamma(caixas, gamma, nome, descricao)
            if resultado:
                resultados.append(resultado)
        
        return resultados
    
    def executar_todos_perturbacoes(self):
        """Executa testes com perturbações ao redor de 0.05"""
        
        # Gammas a testar (log-scale ao redor de 0.05)
        gammas_base = [
            # Valores muito pequenos
            0.001, 0.005, 0.01, 0.02,
            # Região crítica ao redor de 0.05
            0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07,
            # Valores maiores para referência
            0.08, 0.09, 0.1, 0.12, 0.15, 0.2
        ]
        
        # Números de caixas por cenário (já validados)
        cenarios = [
            ("Cúbicas 1x1x1", self.criar_caixas_cubicas(25, 1), "25 caixas cúbicas"),
            ("Altas 1x1x2", self.criar_caixas_1x1x2(25), "12 caixas altas"),
            ("Largas 1x2x1", self.criar_caixas_1x2x1(12), "12 caixas largas"),
            ("Longas 2x1x1", self.criar_caixas_2x1x1(12), "12 caixas longas"),
            ("Aleatórios 1-3", self.criar_caixas_aleatorias(8, 1, 3), "8 caixas aleatórias"),
        ]
        
        todos_resultados = []
        
        for nome, criador, desc in cenarios:
            resultados = self.executar_para_cenario(criador, nome, desc, gammas_base)
            todos_resultados.append({
                'cenario': nome,
                'resultados': resultados
            })
            
            # Mostrar melhores deste cenário
            if resultados:
                melhores = sorted(resultados, key=lambda x: x['fitness_mean'], reverse=True)[:3]
                print(f"\n  🏆 Melhores para {nome}:")
                for i, r in enumerate(melhores, 1):
                    print(f"     {i}. γ={r['gamma']:.4f} → {r['fitness_mean']:.6f} ± {r['fitness_std']:.6f}")
        
        # Salvar e analisar
        self.salvar_resultados(todos_resultados)
        self.analisar_tendencias(todos_resultados)
        
        return todos_resultados
    
    def salvar_resultados(self, todos_resultados):
        """Salva todos os resultados"""
        
        # Salvar JSON
        with open(os.path.join(self.pasta_testes, 'resultados_perturbacao.json'), 'w') as f:
            json.dump(todos_resultados, f, indent=2)
        
        # Criar CSV comparativo
        with open(os.path.join(self.pasta_testes, 'comparativo_gammas.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cenario', 'gamma', 'fitness_mean', 'fitness_std', 'fitness_min', 'fitness_max', 'tempo', 'sucesso'])
            
            for cenario in todos_resultados:
                for r in cenario['resultados']:
                    writer.writerow([
                        cenario['cenario'],
                        r['gamma'],
                        f"{r['fitness_mean']:.6f}",
                        f"{r['fitness_std']:.6f}",
                        f"{r['fitness_min']:.6f}",
                        f"{r['fitness_max']:.6f}",
                        f"{r['tempo_mean']:.2f}",
                        f"{r['sucesso']:.0%}"
                    ])
        
        print(f"\n📁 Resultados salvos em: {self.pasta_testes}/")
    
    def analisar_tendencias(self, todos_resultados):
        """Analisa tendências e encontra o γ ótimo global"""
        
        print("\n" + "="*80)
        print("ANÁLISE DE TENDÊNCIAS")
        print("="*80)
        
        # Coletar todos os gammas e seus fitness médios por cenário
        todos_gammas = {}
        
        for cenario in todos_resultados:
            for r in cenario['resultados']:
                gamma = r['gamma']
                if gamma not in todos_gammas:
                    todos_gammas[gamma] = []
                todos_gammas[gamma].append(r['fitness_mean'])
        
        # Calcular média global por gamma
        media_global = {}
        for gamma, fitnesses in todos_gammas.items():
            media_global[gamma] = np.mean(fitnesses)
        
        # Ordenar por fitness
        ordenados = sorted(media_global.items(), key=lambda x: x[1], reverse=True)
        
        print("\n📊 MÉDIA GLOBAL POR GAMMA (todos os cenários):")
        print("-"*50)
        print(f"{'Gamma':<12} {'Fitness Médio':<18} {'Rank':<6}")
        print("-"*50)
        
        for i, (gamma, fitness) in enumerate(ordenados[:10], 1):
            destaque = "🏆" if i == 1 else "  "
            print(f"{destaque} {gamma:<12} {fitness:+.6f}       #{i}")
        
        # Encontrar o melhor gamma
        melhor_gamma, melhor_fitness = ordenados[0]
        
        print("\n" + "="*80)
        print("🎯 CONCLUSÃO")
        print("="*80)
        print(f"\n✅ Melhor γ global: {melhor_gamma}")
        print(f"   Fitness médio: {melhor_fitness:.6f}")
        
        # Verificar se 0.05 é realmente o melhor
        gamma_005_fitness = media_global.get(0.05, None)
        if gamma_005_fitness:
            diferenca = (melhor_fitness - gamma_005_fitness) / abs(gamma_005_fitness) * 100
            if abs(diferenca) < 1:
                print(f"\n📌 γ=0.05 está entre os melhores (diferença de {diferenca:.2f}%)")
            else:
                print(f"\n⚠️ γ=0.05 NÃO é o melhor! Diferença de {diferenca:.2f}%")
                print(f"   Melhor é γ={melhor_gamma}")
        
        # Salvar análise
        with open(os.path.join(self.pasta_testes, 'analise_tendencias.txt'), 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ANÁLISE DE TENDÊNCIAS - PERTURBAÇÕES\n")
            f.write("="*80 + "\n\n")
            
            f.write("MÉDIA GLOBAL POR GAMMA:\n")
            for gamma, fitness in ordenados:
                f.write(f"  γ={gamma:.4f}: {fitness:+.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"MELHOR GAMMA GLOBAL: {melhor_gamma}\n")
            f.write(f"Fitness: {melhor_fitness:.6f}\n")


def main():
    L, W, H = 4, 5, 6
    
    print("="*80)
    print("TESTES DE PERTURBAÇÃO AO REDOR DO γ ÓTIMO (0.05)")
    print("="*80)
    print(f"Container: {L}x{W}x{H} = {L*W*H}")
    print(f"Execuções por gamma: 5")
    print(f"Gerações por teste: 100")
    print(f"OBS: Violação em X é PROIBIDA (restrição rígida)")
    print("="*80)
    
    testador = TestePerturbacao(
        L=L, W=W, H=H,
        num_execucoes=5,
        geracoes_por_teste=100
    )
    
    testador.executar_todos_perturbacoes()
    
    print("\n" + "="*80)
    print("✅ TESTES CONCLUÍDOS!")
    print("="*80)


if __name__ == "__main__":
    main()