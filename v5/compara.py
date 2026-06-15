# comparador_configuracoes.py
"""
Compara diferentes configurações lado a lado
Mede violações de ordem reais
"""

import json
import os
from brkga_3d import BRKGA, DecodificadorConstrutivo

class ComparadorConfiguracoes:
    def __init__(self, caixas, L, W, H):
        self.caixas = caixas
        self.L = L
        self.W = W
        self.H = H
    
    def contar_violacoes(self, solucao):
        """Conta violações de ordem na solução"""
        if not solucao:
            return 0, 0, 0
        
        violacoes_x = 0
        violacoes_y = 0
        violacoes_z = 0
        total_pares = 0
        
        # Reconstruir posições
        posicoes = []
        for i, pos in enumerate(solucao):
            if pos:
                x, y, z, r = pos
                caixa = self.caixas[i]
                posicoes.append({
                    'indice': i,
                    'nome': caixa[5],
                    'prioridade': caixa[4],
                    'posicao': (x, y, z, r)
                })
        
        # Contar violações
        for i, p1 in enumerate(posicoes):
            for j, p2 in enumerate(posicoes):
                if i == j:
                    continue
                if p1['prioridade'] < p2['prioridade']:
                    total_pares += 1
                    if p1['posicao'][0] > p2['posicao'][0]:
                        violacoes_x += 1
                    if abs(p1['posicao'][1] - p2['posicao'][1]) > 0:
                        violacoes_y += 1
                    if p1['posicao'][2] < p2['posicao'][2]:
                        violacoes_z += 1
        
        return violacoes_x, violacoes_y, violacoes_z, total_pares
    
    def testar_configuracao(self, nome, **kwargs):
        """Testa uma configuração e retorna estatísticas"""
        print(f"\n{'='*50}")
        print(f"Testando: {nome}")
        print(f"{'='*50}")
        
        brkga = BRKGA(
            num_genes=len(self.caixas)*2,
            num_individuos=50,
            num_elite=10,
            num_mutantes=5,
            prob_crossover=0.7,
            caixas=self.caixas,
            L=self.L, W=self.W, H=self.H,
            verbose=False,
            **kwargs
        )
        
        solucao, fitness, _, _ = brkga.executar_continuo(
            geracoes_max=50,
            sem_melhoria_max=20
        )
        
        if solucao:
            vx, vy, vz, total = self.contar_violacoes(solucao)
            return {
                'nome': nome,
                'fitness': fitness,
                'violacoes_x': vx,
                'violacoes_y': vy,
                'violacoes_z': vz,
                'total_violacoes': vx + vy + vz,
                'total_pares': total,
                'taxa_violacao': (vx + vy + vz) / total if total > 0 else 0
            }
        return None
    
    def comparar_todos(self):
        """Compara todas as configurações relevantes"""
        
        configuracoes = [
            # Único - diferentes gammas
            ('Único γ=0.17', {'gamma_unico': 0.17}),
            ('Único γ=0.30', {'gamma_unico': 0.30}),
            ('Único γ=0.50', {'gamma_unico': 0.50}),
            ('Único γ=1.00', {'gamma_unico': 1.00}),
            
            # Híbrido - priorizando diferentes eixos
            ('Híbrido γ=0.5 px=2.0 (prioriza X)', 
             {'gamma_unico': 0.5, 'peso_x': 2.0, 'peso_y': 0.5, 'peso_z': 0.5}),
            ('Híbrido γ=0.5 pz=2.0 (prioriza Z)', 
             {'gamma_unico': 0.5, 'peso_x': 0.5, 'peso_y': 0.5, 'peso_z': 2.0}),
            
            # Separado
            ('Separado gx=2.0 (prioriza X)', 
             {'gamma_x': 2.0, 'gamma_y': 0.5, 'gamma_z': 0.5}),
            ('Separado gz=2.0 (prioriza Z)', 
             {'gamma_x': 0.5, 'gamma_y': 0.5, 'gamma_z': 2.0}),
        ]
        
        resultados = []
        for nome, params in configuracoes:
            resultado = self.testar_configuracao(nome, **params)
            if resultado:
                resultados.append(resultado)
        
        self.mostrar_comparacao(resultados)
        return resultados
    
    def mostrar_comparacao(self, resultados):
        """Mostra tabela comparativa"""
        
        print("\n" + "="*80)
        print("COMPARAÇÃO ENTRE CONFIGURAÇÕES")
        print("="*80)
        
        # Cabeçalho
        print(f"\n{'Configuração':<35} {'Fitness':<12} {'Viol X':<8} {'Viol Y':<8} {'Viol Z':<8} {'Total':<8} {'Taxa':<8}")
        print("-"*80)
        
        # Ordenar por total de violações (menos é melhor)
        resultados.sort(key=lambda x: x['total_violacoes'])
        
        for r in resultados:
            print(f"{r['nome']:<35} {r['fitness']:+.4f}     "
                  f"{r['violacoes_x']:<8} {r['violacoes_y']:<8} {r['violacoes_z']:<8} "
                  f"{r['total_violacoes']:<8} {r['taxa_violacao']:.1%}")
        
        # Análise
        print("\n" + "="*80)
        print("ANÁLISE E RECOMENDAÇÃO")
        print("="*80)
        
        # Encontrar melhor para cada critério
        menos_violacoes = min(resultados, key=lambda x: x['total_violacoes'])
        melhor_fitness = max(resultados, key=lambda x: x['fitness'])
        
        print(f"\n✅ MENOS VIOLAÇÕES: {menos_violacoes['nome']}")
        print(f"   Total: {menos_violacoes['total_violacoes']} violações")
        print(f"   Fitness: {menos_violacoes['fitness']:.4f}")
        
        print(f"\n✅ MELHOR FITNESS: {melhor_fitness['nome']}")
        print(f"   Fitness: {melhor_fitness['fitness']:.4f}")
        print(f"   Violações: {melhor_fitness['total_violacoes']}")
        
        # Recomendação baseada em trade-off
        print("\n" + "="*80)
        print("🎯 RECOMENDAÇÃO")
        print("="*80)
        
        if menos_violacoes['nome'] == melhor_fitness['nome']:
            print(f"\nUse: {menos_violacoes['nome']}")
            print("(Melhor em ambos os critérios!)")
        else:
            print("\nESCOLHA BASEADA NA SUA PRIORIDADE:")
            print(f"  • Se PRIORIDADE é evitar violações → {menos_violacoes['nome']}")
            print(f"  • Se PRIORIDADE é maximizar fitness → {melhor_fitness['nome']}")
            print(f"  • Se quer EQUILÍBRIO → Use γ=0.5 (Único)")


def main():
    # Suas caixas
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
    
    comparador = ComparadorConfiguracoes(caixas, L, W, H)
    resultados = comparador.comparar_todos()


if __name__ == "__main__":
    main()