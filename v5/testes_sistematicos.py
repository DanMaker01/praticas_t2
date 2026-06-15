"""
Testes refinados para encontrar o γ ótimo
Foco na região entre 0.3 e 0.8 onde está o melhor valor
"""

import json
import os
import time
import random
import numpy as np
from datetime import datetime
import csv
from brkga_3d import BRKGA

class TestesRefinados:
    """Testes mais precisos para encontrar o γ ótimo"""
    
    def __init__(self, caixas, L, W, H, num_execucoes=5, geracoes_por_teste=50):
        self.caixas = caixas
        self.L = L
        self.W = W
        self.H = H
        self.num_execucoes = num_execucoes
        self.geracoes_por_teste = geracoes_por_teste
        self.resultados = []
        
        self.pasta_testes = f"testes_refinados_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
    
    def executar_gamma(self, gamma):
        """Executa testes para um valor específico de gamma"""
        
        print(f"\n{'='*50}")
        print(f"Testando γ = {gamma}")
        print(f"{'='*50}")
        
        resultados_gamma = []
        
        for execucao in range(self.num_execucoes):
            print(f"  Execução {execucao+1}/{self.num_execucoes}", end=" ", flush=True)
            
            brkga = BRKGA(
                num_genes=len(self.caixas)*2,
                num_individuos=50,
                num_elite=10,
                num_mutantes=5,
                prob_crossover=0.7,
                caixas=self.caixas,
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
            
            resultados_gamma.append({
                'execucao': execucao + 1,
                'fitness': fitness,
                'tempo': tempo,
                'geracoes': len(historico),
                'volume': brkga.volume_ocupado
            })
            
            print(f"→ Fitness: {fitness:.6f}")
        
        # Estatísticas
        fitnesses = [r['fitness'] for r in resultados_gamma]
        
        return {
            'gamma': gamma,
            'resultados': resultados_gamma,
            'stats': {
                'fitness_mean': np.mean(fitnesses),
                'fitness_std': np.std(fitnesses),
                'fitness_max': max(fitnesses),
                'fitness_min': min(fitnesses),
                'fitness_median': np.median(fitnesses),
                'tempo_mean': np.mean([r['tempo'] for r in resultados_gamma]),
                'volume_mean': np.mean([r['volume'] for r in resultados_gamma])
            }
        }
    
    def executar_primeira_varredura(self):
        """Primeira varredura: região ampla ao redor de 0.5"""
        
        print("\n" + "="*60)
        print("PRIMEIRA VARREDURA - Região 0.3 a 0.8")
        print("="*60)
        
        # Gammas mais próximos de 0.5
        gammas = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        
        for gamma in gammas:
            resultado = self.executar_gamma(gamma)
            self.resultados.append(resultado)
            self.salvar_parcial()
        
        # Encontrar o melhor até agora
        melhor = max(self.resultados, key=lambda x: x['stats']['fitness_mean'])
        print(f"\n📍 Melhor até agora: γ = {melhor['gamma']} com fitness {melhor['stats']['fitness_mean']:.6f}")
        
        return melhor['gamma']
    
    def executar_segunda_varredura(self, centro, raio=0.1, passos=7):
        """Segunda varredura: foco na região do melhor"""
        
        print("\n" + "="*60)
        print(f"SEGUNDA VARREDURA - Região ao redor de {centro}")
        print("="*60)
        
        # Criar gammas ao redor do centro
        inicio = max(0.1, centro - raio)
        fim = min(2.0, centro + raio)
        gammas = np.linspace(inicio, fim, passos).tolist()
        
        # Remover duplicatas próximas
        gammas = [round(g, 3) for g in gammas]
        
        for gamma in gammas:
            # Verificar se já não testamos este valor
            if any(r['gamma'] == gamma for r in self.resultados):
                print(f"γ = {gamma} já testado, pulando...")
                continue
            
            resultado = self.executar_gamma(gamma)
            self.resultados.append(resultado)
            self.salvar_parcial()
        
        # Encontrar o melhor
        melhor = max(self.resultados, key=lambda x: x['stats']['fitness_mean'])
        return melhor['gamma']
    
    def executar_varredura_final(self, centro, raio=0.03, passos=5):
        """Terceira varredura: ajuste fino"""
        
        print("\n" + "="*60)
        print(f"VARREDURA FINAL - Ajuste fino ao redor de {centro}")
        print("="*60)
        
        inicio = max(0.1, centro - raio)
        fim = centro + raio
        gammas = np.linspace(inicio, fim, passos).tolist()
        gammas = [round(g, 3) for g in gammas]
        
        for gamma in gammas:
            if any(r['gamma'] == gamma for r in self.resultados):
                print(f"γ = {gamma} já testado, pulando...")
                continue
            
            resultado = self.executar_gamma(gamma)
            self.resultados.append(resultado)
            self.salvar_parcial()
        
        # Melhor final
        melhor = max(self.resultados, key=lambda x: x['stats']['fitness_mean'])
        return melhor
    
    def salvar_parcial(self):
        """Salva resultados parciais"""
        
        arquivo = os.path.join(self.pasta_testes, 'resultados_parciais.json')
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2)
    
    def salvar_resultados_finais(self):
        """Salva resultados finais com análises"""
        
        # Ordenar por fitness
        self.resultados.sort(key=lambda x: x['stats']['fitness_mean'], reverse=True)
        
        # 1. JSON completo
        with open(os.path.join(self.pasta_testes, 'resultados_completos.json'), 'w') as f:
            json.dump(self.resultados, f, indent=2)
        
        # 2. CSV comparativo
        with open(os.path.join(self.pasta_testes, 'comparativo_gammas.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gamma', 'fitness_mean', 'fitness_std', 'fitness_max', 'fitness_min', 
                           'tempo_mean', 'volume', 'ranking'])
            
            for i, r in enumerate(self.resultados, 1):
                writer.writerow([
                    r['gamma'],
                    f"{r['stats']['fitness_mean']:.6f}",
                    f"{r['stats']['fitness_std']:.6f}",
                    f"{r['stats']['fitness_max']:.6f}",
                    f"{r['stats']['fitness_min']:.6f}",
                    f"{r['stats']['tempo_mean']:.2f}",
                    r['stats']['volume_mean'],
                    i
                ])
        
        # 3. Relatório detalhado
        with open(os.path.join(self.pasta_testes, 'relatorio_final.txt'), 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO FINAL - OTIMIZAÇÃO DO GAMMA\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data: {datetime.now().isoformat()}\n")
            f.write(f"Container: {self.L}x{self.W}x{self.H}\n")
            f.write(f"Caixas: {len(self.caixas)}\n")
            f.write(f"Execuções por gamma: {self.num_execucoes}\n")
            f.write(f"Gerações por teste: {self.geracoes_por_teste}\n\n")
            
            f.write("RESULTADOS ORDENADOS POR FITNESS:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Rank':<6} {'Gamma':<10} {'Fitness Médio':<18} {'Desvio':<12} {'Volume':<8}\n")
            f.write("-"*70 + "\n")
            
            for i, r in enumerate(self.resultados, 1):
                f.write(f"{i:<6} {r['gamma']:<10} {r['stats']['fitness_mean']:+.6f}     "
                       f"±{r['stats']['fitness_std']:.6f}   {r['stats']['volume_mean']:.0f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMENDAÇÃO FINAL\n")
            f.write("="*70 + "\n")
            
            melhor = self.resultados[0]
            f.write(f"\n🏆 MELHOR GAMMA ENCONTRADO: {melhor['gamma']}\n")
            f.write(f"   Fitness médio: {melhor['stats']['fitness_mean']:.6f}\n")
            f.write(f"   Desvio padrão: {melhor['stats']['fitness_std']:.6f}\n")
            f.write(f"   Volume: {melhor['stats']['volume_mean']:.0f}\n\n")
            
            f.write("Configuração recomendada:\n")
            f.write("-"*50 + "\n")
            f.write(f"gamma_unico = {melhor['gamma']}\n")
            f.write("exigir_estabilidade_descarga = True\n")
            f.write("suporte_minimo = 1.0\n")
        
        print(f"\n📁 Resultados salvos em: {self.pasta_testes}/")
    
    def mostrar_resultados(self):
        """Mostra resultados formatados no console"""
        
        print("\n" + "="*70)
        print("RESULTADOS DOS TESTES REFINADOS")
        print("="*70)
        
        # Ordenar por fitness
        self.resultados.sort(key=lambda x: x['stats']['fitness_mean'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Gamma':<10} {'Fitness Médio':<18} {'Desvio':<14} {'Volume':<8}")
        print("-"*70)
        
        for i, r in enumerate(self.resultados, 1):
            prefixo = "🏆 " if i == 1 else "   "
            print(f"{prefixo}{i:<6} {r['gamma']:<10} {r['stats']['fitness_mean']:+.6f}     "
                  f"±{r['stats']['fitness_std']:.6f}   {r['stats']['volume_mean']:.0f}")
        
        # Melhor configuração
        melhor = self.resultados[0]
        segundo = self.resultados[1] if len(self.resultados) > 1 else None
        
        print("\n" + "="*70)
        print("📊 ANÁLISE ESTATÍSTICA")
        print("="*70)
        
        print(f"\nMelhor γ: {melhor['gamma']}")
        print(f"  Fitness: {melhor['stats']['fitness_mean']:.6f} ± {melhor['stats']['fitness_std']:.6f}")
        
        if segundo:
            diferenca = (segundo['stats']['fitness_mean'] - melhor['stats']['fitness_mean']) / abs(melhor['stats']['fitness_mean']) * 100
            print(f"\nDiferença para o 2º lugar (γ={segundo['gamma']}): {diferenca:.2f}%")
            
            if diferenca < 5:
                print("  ⚠️ Diferença pequena - múltiplos gammas são igualmente bons")
            else:
                print("  ✅ Diferença significativa - este gamma é claramente superior")
        
        # Verificar consistência
        print(f"\nConsistência (desvio padrão):")
        if melhor['stats']['fitness_std'] < 0.0005:
            print("  ✅ Excelente - resultados muito consistentes")
        elif melhor['stats']['fitness_std'] < 0.001:
            print("  👍 Bom - resultados consistentes")
        else:
            print("  ⚠️ Variável - considere aumentar o número de execuções")


def main():
    # Suas caixas (mesmas do teste anterior)
    caixas = [
        (0, 1, 1, 1, 1, "Caixa_A1"),
        (1, 1, 1, 5, 1, "Caixa_A2"),
        (2, 2, 1, 1, 2, "Caixa_B1"),
        (3, 1, 1, 3, 2, "Caixa_B2"),
        (4, 2, 2, 3, 3, "Caixa_C1"),
        (5, 2, 2, 1, 3, "Caixa_C2"),
        (6, 2, 2, 1, 3, "Caixa_C3"),
        (7, 2, 2, 1, 4, "Caixa_D1"),
        (8, 2, 2, 1, 4, "Caixa_D2"),
        (9, 2, 2, 1, 5, "Caixa_E1"),
        (10, 2, 2, 1, 5, "Caixa_E2"),
        (11, 2, 2, 1, 5, "Caixa_E3"),
    ]
    
    L, W, H = 4, 5, 6
    
    print("\n" + "="*70)
    print("TESTES REFINADOS PARA ENCONTRAR O γ ÓTIMO")
    print("="*70)
    print(f"Container: {L}x{W}x{H}")
    print(f"Caixas: {len(caixas)}")
    print(f"Volume máximo possível: {sum(c[1]*c[2]*c[3] for c in caixas)}")
    print(f"Execuções por gamma: 5")
    print(f"Gerações por teste: 50")
    print("="*70)
    
    testador = TestesRefinados(
        caixas=caixas,
        L=L, W=W, H=H,
        num_execucoes=5,      # 5 repetições para maior precisão
        geracoes_por_teste=50 # 50 gerações para melhor convergência
    )
    
    # Fase 1: Varredura inicial (0.3 a 0.8)
    melhor_gamma = testador.executar_primeira_varredura()
    
    # Fase 2: Foco na região do melhor (raio 0.1)
    melhor_gamma = testador.executar_segunda_varredura(melhor_gamma, raio=0.1, passos=7)
    
    # Fase 3: Ajuste fino (raio 0.03)
    resultado_final = testador.executar_varredura_final(melhor_gamma, raio=0.03, passos=5)
    
    # Salvar e mostrar resultados
    testador.salvar_resultados_finais()
    testador.mostrar_resultados()
    
    print("\n" + "="*70)
    print("✅ TESTES CONCLUÍDOS!")
    print("="*70)
    print(f"\n🏆 GAMMA ÓTIMO RECOMENDADO: {resultado_final['gamma']}")
    print(f"   Fitness esperado: {resultado_final['stats']['fitness_mean']:.6f}")
    print(f"\nUse esta configuração no seu BRKGA:")
    print(f"    gamma_unico = {resultado_final['gamma']}")


if __name__ == "__main__":
    main()