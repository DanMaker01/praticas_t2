"""
Teste de Escalabilidade do BRKGA 3D - Com caixas cubicas 1x1x1
Aumenta progressivamente o numero de caixas e dimensiona o container
"""

import random
import numpy as np
import json
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brkga_3d import BRKGA

class TesteEscalabilidade:
    """Testa a escalabilidade do BRKGA com caixas cubicas 1x1x1"""
    
    def __init__(self):
        self.resultados = []
        self.pasta_testes = f"testes_escalabilidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
        
        # Parametros fixos do BRKGA (otimizados)
        self.gamma_unico = 0.001   # Gamma otimo encontrado
        self.num_individuos = 50
        self.num_elite = 10
        self.num_mutantes = 5
        self.prob_crossover = 0.7
        self.geracoes_max = 150
        self.sem_melhoria_max = 30
    
    def criar_caixas_cubicas(self, n):
        """Cria N caixas cubicas 1x1x1"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 1, 1, prioridade, f"Cubo_{i}"))
        return caixas
    
    def executar_teste(self, nome_teste, caixas, L, W, H, tempo_max_minutos=5):
        """Executa um unico teste e retorna os resultados"""
        
        print(f"\n{'='*60}")
        print(f"TESTE: {nome_teste}")
        print(f"Container: {L}x{W}x{H} = {L*W*H}")
        print(f"Caixas: {len(caixas)} (1x1x1)")
        print(f"Ocupacao teorica: {len(caixas)/(L*W*H)*100:.1f}%")
        print(f"Tempo maximo: {tempo_max_minutos} minutos")
        print(f"{'='*60}")
        
        brkga = BRKGA(
            num_genes=len(caixas)*2,
            num_individuos=self.num_individuos,
            num_elite=self.num_elite,
            num_mutantes=self.num_mutantes,
            prob_crossover=self.prob_crossover,
            caixas=caixas,
            L=L, W=W, H=H,
            gamma_unico=self.gamma_unico,
            exigir_estabilidade_descarga=True,
            suporte_minimo=1.0,
            verbose=True
        )
        
        inicio = time.time()
        solucao, fitness, historico, motivo = brkga.executar_continuo(
            geracoes_max=self.geracoes_max,
            sem_melhoria_max=self.sem_melhoria_max,
            tempo_maximo_horas=tempo_max_minutos/60
        )
        tempo_total = time.time() - inicio
        
        resultado = {
            'nome_teste': nome_teste,
            'L': L, 'W': W, 'H': H,
            'volume_container': L*W*H,
            'num_caixas': len(caixas),
            'volume_caixas': len(caixas),
            'ocupacao_teorica': len(caixas)/(L*W*H)*100,
            'volume_ocupado': brkga.volume_ocupado,
            'utilizacao_real': brkga.volume_ocupado / (L*W*H) * 100,
            'fitness': fitness,
            'tempo_total': tempo_total,
            'geracoes_executadas': brkga.geracao_atual,
            'solucoes_invalidas': brkga.solucoes_invalidas,
            'motivo_parada': motivo,
            'sucesso': fitness != -float('inf')
        }
        
        self.resultados.append(resultado)
        self.salvar_resultado_individual(resultado)
        
        return resultado
    
    def salvar_resultado_individual(self, resultado):
        """Salva resultado individual em JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.pasta_testes}/teste_{resultado['nome_teste']}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2)
    
    def salvar_resultados_completos(self):
        """Salva todos os resultados em CSV e JSON"""
        df = pd.DataFrame(self.resultados)
        
        csv_file = f"{self.pasta_testes}/resultados_completos.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        json_file = f"{self.pasta_testes}/resultados_completos.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2)
        
        print(f"\nResultados salvos em: {self.pasta_testes}/")
        return df
    
    def gerar_relatorio(self):
        """Gera relatorio com graficos de escalabilidade"""
        
        df = pd.DataFrame(self.resultados)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Tempo vs Numero de caixas
        ax1 = axes[0, 0]
        for nome in df['nome_teste'].unique():
            subset = df[df['nome_teste'] == nome]
            ax1.plot(subset['num_caixas'], subset['tempo_total'], 'o-', label=nome, linewidth=2, markersize=8)
        ax1.set_xlabel('Numero de Caixas')
        ax1.set_ylabel('Tempo Total (segundos)')
        ax1.set_title('Escalabilidade: Tempo vs Quantidade de Caixas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Utilizacao vs Ocupacao teorica
        ax2 = axes[0, 1]
        ax2.scatter(df['ocupacao_teorica'], df['utilizacao_real'], s=100, alpha=0.6, c='blue')
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Ideal (100%)')
        ax2.set_xlabel('Ocupacao Teorica (%)')
        ax2.set_ylabel('Utilizacao Real (%)')
        ax2.set_title('Eficiencia de Empacotamento')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fitness vs Numero de caixas
        ax3 = axes[1, 0]
        for nome in df['nome_teste'].unique():
            subset = df[df['nome_teste'] == nome]
            ax3.plot(subset['num_caixas'], subset['fitness'], 'o-', label=nome, linewidth=2, markersize=8)
        ax3.set_xlabel('Numero de Caixas')
        ax3.set_ylabel('Fitness')
        ax3.set_title('Qualidade da Solucao vs Quantidade de Caixas')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Taxa de sucesso
        ax4 = axes[1, 1]
        df_sucesso = df.groupby('nome_teste').agg({'sucesso': 'mean', 'num_caixas': 'first'}).reset_index()
        cores = ['green' if s else 'red' for s in df_sucesso['sucesso']]
        ax4.bar(df_sucesso['nome_teste'], df_sucesso['sucesso'] * 100, color=cores, alpha=0.7)
        ax4.set_ylabel('Taxa de Sucesso (%)')
        ax4.set_title('Taxa de Sucesso por Teste')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        fig_file = f"{self.pasta_testes}/escalabilidade_graficos.png"
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        print(f"\nGraficos salvos em: {fig_file}")
        plt.show()
        
        return fig
    
    def imprimir_resumo(self):
        """Imprime resumo dos testes"""
        
        print("\n" + "="*80)
        print("RESUMO DOS TESTES DE ESCALABILIDADE (CAIXAS CUBICAS 1x1x1)")
        print("="*80)
        
        df = pd.DataFrame(self.resultados)
        
        print(f"\nESTATISTICAS GERAIS:")
        print(f"   Total de testes: {len(df)}")
        print(f"   Testes bem-sucedidos: {df['sucesso'].sum()}")
        print(f"   Taxa de sucesso geral: {df['sucesso'].mean()*100:.1f}%")
        
        print(f"\nDETALHES POR TESTE:")
        print("-"*80)
        for _, r in df.iterrows():
            status = "OK" if r['sucesso'] else "FALHA"
            print(f"{status} {r['nome_teste']:20} | Caixas: {r['num_caixas']:3} | "
                  f"Ocup: {r['ocupacao_teorica']:5.1f}% | Util: {r['utilizacao_real']:5.1f}% | "
                  f"Fitness: {r['fitness']:.6f} | Tempo: {r['tempo_total']:5.2f}s")
        
        df_sucesso = df[df['sucesso']]
        if not df_sucesso.empty:
            melhor = df_sucesso.loc[df_sucesso['fitness'].idxmax()]
            print(f"\nMELHOR RESULTADO:")
            print(f"   Teste: {melhor['nome_teste']}")
            print(f"   Caixas: {melhor['num_caixas']}")
            print(f"   Fitness: {melhor['fitness']:.6f}")
            print(f"   Utilizacao: {melhor['utilizacao_real']:.1f}%")


def main():
    print("\n" + "="*80)
    print("TESTE DE ESCALABILIDADE - BRKGA 3D (CAIXAS CUBICAS 1x1x1)")
    print("="*80)
    print("Gamma = 0.001 (otimo encontrado)")
    print("="*80)
    
    testador = TesteEscalabilidade()
    
    # FASE 1: Container pequeno (4x5x6)
    # Limite teorico: 120 caixas
    print("\n" + "FASE 1: CONTAINER PEQUENO (4x5x6)")
    print("-"*40)
    
    L, W, H = 4, 5, 6
    
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        caixas = testador.criar_caixas_cubicas(n)
        resultado = testador.executar_teste(f"Pequeno_{n}", caixas, L, W, H, tempo_max_minutos=3)
        
        if not resultado['sucesso']:
            print(f"\nFalha detectada em {n} caixas. Parando Fase 1.")
            break
    
    # FASE 2: Container medio (8x10x12)
    # Limite teorico: 960 caixas
    print("\n" + "FASE 2: CONTAINER MEDIO (8x10x12)")
    print("-"*40)
    
    L, W, H = 8, 10, 12
    
    for n in [50, 100, 150, 200, 250, 300, 350, 400]:
        caixas = testador.criar_caixas_cubicas(n)
        resultado = testador.executar_teste(f"Medio_{n}", caixas, L, W, H, tempo_max_minutos=5)
        
        if not resultado['sucesso']:
            print(f"\nFalha detectada em {n} caixas. Parando Fase 2.")
            break
    
    # FASE 3: Container grande (10x12x15)
    # Limite teorico: 1800 caixas
    print("\n" + "FASE 3: CONTAINER GRANDE (10x12x15)")
    print("-"*40)
    
    L, W, H = 10, 12, 15
    
    for n in [100, 200, 300, 400, 500, 600, 700, 800]:
        caixas = testador.criar_caixas_cubicas(n)
        resultado = testador.executar_teste(f"Grande_{n}", caixas, L, W, H, tempo_max_minutos=8)
        
        if not resultado['sucesso']:
            print(f"\nFalha detectada em {n} caixas. Parando Fase 3.")
            break
    
    # FASE 4: Container extra grande (12x15x18)
    # Limite teorico: 3240 caixas
    print("\n" + "FASE 4: CONTAINER EXTRA GRANDE (12x15x18)")
    print("-"*40)
    
    L, W, H = 12, 15, 18
    
    for n in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        caixas = testador.criar_caixas_cubicas(n)
        resultado = testador.executar_teste(f"Extra_{n}", caixas, L, W, H, tempo_max_minutos=10)
        
        if not resultado['sucesso']:
            print(f"\nFalha detectada em {n} caixas. Parando Fase 4.")
            break
    
    # RELATORIO FINAL
    print("\n" + "GERANDO RELATORIO FINAL")
    print("-"*40)
    
    testador.salvar_resultados_completos()
    testador.gerar_relatorio()
    testador.imprimir_resumo()
    
    print("\n" + "="*80)
    print("TESTES DE ESCALABILIDADE CONCLUIDOS")
    print("="*80)


if __name__ == "__main__":
    main()