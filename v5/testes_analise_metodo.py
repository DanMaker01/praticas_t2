"""
Analise de Escalabilidade e Convergencia do BRKGA 3D
Estuda: tempo de resolucao, convergencia, impacto da heterogeneidade
"""

import json
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from brkga_3d import BRKGA

class AnaliseEscalabilidade:
    """Analisa escalabilidade do BRKGA para diferentes cenarios"""
    
    def __init__(self):
        self.resultados = []
        self.pasta_testes = f"analise_escalabilidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
        
        # Configuracao base (otimizada)
        self.gamma_z = 0.001
        self.num_individuos = 50
        self.num_elite = 10
        self.num_mutantes = 5
        self.prob_crossover = 0.7
    
    def gerar_caixas_homogeneas(self, n, tamanho=1):
        """Gera caixas homogeneas (todas iguais)"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, tamanho, tamanho, tamanho, prioridade, f"Hom_{i}"))
        return caixas
    
    def gerar_caixas_heterogeneas(self, n, min_dim=1, max_dim=3):
        """Gera caixas heterogeneas (dimensoes variadas)"""
        caixas = []
        for i in range(n):
            l = random.randint(min_dim, max_dim)
            w = random.randint(min_dim, max_dim)
            h = random.randint(min_dim, max_dim)
            prioridade = (i % 5) + 1
            caixas.append((i, l, w, h, prioridade, f"Heter_{i}"))
        return caixas
    
    def executar_teste(self, nome, caixas, L, W, H, max_geracoes):
        """Executa um teste ate convergencia ou max_geracoes"""
        
        print(f"  Executando: {nome} (max_geracoes={max_geracoes})", end=" ", flush=True)
        
        brkga = BRKGA(
            num_genes=len(caixas)*2,
            num_individuos=self.num_individuos,
            num_elite=self.num_elite,
            num_mutantes=self.num_mutantes,
            prob_crossover=self.prob_crossover,
            caixas=caixas,
            L=L, W=W, H=H,
            gamma_x=0.0, gamma_y=0.0, gamma_z=self.gamma_z,
            exigir_estabilidade_descarga=True,
            suporte_minimo=1.0,
            verbose=False
        )
        
        inicio = time.time()
        solucao, fitness, historico, motivo = brkga.executar_continuo(
            geracoes_max=max_geracoes,
            sem_melhoria_max=max_geracoes // 2,
            tempo_maximo_horas=0.5
        )
        tempo = time.time() - inicio
        
        # Analisar convergencia
        geracao_convergencia = self.encontrar_convergencia(historico)
        
        resultado = {
            'nome': nome,
            'n_caixas': len(caixas),
            'tipo': 'homogeneo' if 'Hom' in nome else 'heterogeneo',
            'max_geracoes': max_geracoes,
            'geracoes_executadas': brkga.geracao_atual,
            'geracao_convergencia': geracao_convergencia,
            'fitness': fitness,
            'tempo': tempo,
            'motivo': motivo,
            'sucesso': fitness != -float('inf')
        }
        
        self.resultados.append(resultado)
        print(f"-> {resultado['geracoes_executadas']} ger, {tempo:.2f}s, convergiu em {geracao_convergencia}")
        
        return resultado
    
    def encontrar_convergencia(self, historico, tolerancia=1e-6):
        """Encontra a geracao onde o fitness estabilizou"""
        
        if len(historico) < 10:
            return len(historico)
        
        # Procura onde a variacao e muito pequena
        for i in range(len(historico) - 10, 0, -10):
            if max(historico[i:]) - min(historico[i:]) > tolerancia:
                return i + 10
        
        return len(historico)
    
    def estudar_convergencia(self):
        """Estuda quantas geracoes sao necessarias para convergencia"""
        
        print("\n" + "="*60)
        print("1. ESTUDO DE CONVERGENCIA")
        print("="*60)
        
        L, W, H = 8, 10, 12
        
        resultados_conv = []
        
        for n in [20, 40, 60, 80]:
            caixas = self.gerar_caixas_homogeneas(n, tamanho=1)
            print(f"\nTeste com {n} caixas homogeneas:")
            
            for max_gen in [50, 100, 200, 500]:
                r = self.executar_teste(f"Conv_{n}_{max_gen}", caixas, L, W, H, max_gen)
                resultados_conv.append(r)
        
        return resultados_conv
    
    def estudar_tempo_por_caixa(self):
        """Estuda como o tempo cresce com o numero de caixas"""
        
        print("\n" + "="*60)
        print("2. ESTUDO DE TEMPO POR CAIXA")
        print("="*60)
        
        L, W, H = 8, 10, 12
        max_gen = 200
        
        resultados_tempo = []
        
        # Homogeneas
        print("\nCaixas homogeneas (1x1x1):")
        for n in [10, 20, 30, 40, 50, 60, 70, 80]:
            caixas = self.gerar_caixas_homogeneas(n, tamanho=1)
            r = self.executar_teste(f"Hom_{n}", caixas, L, W, H, max_gen)
            if r['sucesso']:
                resultados_tempo.append(r)
            else:
                print(f"  Falhou em {n} caixas")
                break
        
        # Heterogeneas
        print("\nCaixas heterogeneas (1-3):")
        for n in [10, 20, 30, 40, 50]:
            caixas = self.gerar_caixas_heterogeneas(n, 1, 3)
            r = self.executar_teste(f"Heter_{n}", caixas, L, W, H, max_gen)
            if r['sucesso']:
                resultados_tempo.append(r)
            else:
                print(f"  Falhou em {n} caixas")
                break
        
        return resultados_tempo
    
    def estudar_container_size(self):
        """Estuda o impacto do tamanho do container"""
        
        print("\n" + "="*60)
        print("3. ESTUDO DE TAMANHO DO CONTAINER")
        print("="*60)
        
        n_caixas = 50
        max_gen = 200
        resultados_container = []
        
        containers = [
            (4, 5, 6, "Pequeno"),
            (6, 8, 10, "Medio"),
            (8, 10, 12, "Grande"),
            (10, 12, 15, "Extra"),
        ]
        
        for L, W, H, nome in containers:
            print(f"\nContainer {nome} ({L}x{W}x{H}):")
            caixas = self.gerar_caixas_homogeneas(n_caixas, tamanho=1)
            r = self.executar_teste(f"Container_{nome}", caixas, L, W, H, max_gen)
            resultados_container.append(r)
        
        return resultados_container
    
    def comparar_homogeneo_heterogeneo(self):
        """Compara desempenho entre caixas homogeneas e heterogeneas"""
        
        print("\n" + "="*60)
        print("4. COMPARACAO: HOMOGENEO vs HETEROGENEO")
        print("="*60)
        
        L, W, H = 8, 10, 12
        max_gen = 200
        resultados_comp = []
        
        for n in [20, 30, 40, 50]:
            print(f"\n{n} caixas:")
            
            # Homogeneas
            caixas_hom = self.gerar_caixas_homogeneas(n, tamanho=1)
            r_hom = self.executar_teste(f"Hom_{n}", caixas_hom, L, W, H, max_gen)
            resultados_comp.append(r_hom)
            
            # Heterogeneas
            caixas_het = self.gerar_caixas_heterogeneas(n, 1, 3)
            r_het = self.executar_teste(f"Heter_{n}", caixas_het, L, W, H, max_gen)
            resultados_comp.append(r_het)
            
            if r_hom['sucesso'] and r_het['sucesso']:
                print(f"  Homogeneo: {r_hom['tempo']:.2f}s, Heterogeneo: {r_het['tempo']:.2f}s")
        
        return resultados_comp
    
    def gerar_relatorio(self):
        """Gera graficos e relatorio final"""
        
        df = pd.DataFrame(self.resultados)
        
        if df.empty:
            print("Nenhum resultado para gerar relatorio")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Convergencia: geracoes necessarias vs numero de caixas
        ax1 = axes[0, 0]
        df_conv = df[df['nome'].str.startswith('Conv_')]
        if not df_conv.empty:
            for n in df_conv['n_caixas'].unique():
                subset = df_conv[df_conv['n_caixas'] == n]
                ax1.plot(subset['max_geracoes'], subset['geracao_convergencia'], 'o-', label=f'{n} caixas')
            ax1.set_xlabel('Maximo de Geracoes')
            ax1.set_ylabel('Geracao de Convergencia')
            ax1.set_title('Convergencia: quando o fitness estabiliza')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Tempo vs numero de caixas
        ax2 = axes[0, 1]
        for tipo in ['homogeneo', 'heterogeneo']:
            subset = df[df['tipo'] == tipo]
            if not subset.empty:
                ax2.plot(subset['n_caixas'], subset['tempo'], 'o-', label=tipo, linewidth=2)
        ax2.set_xlabel('Numero de Caixas')
        ax2.set_ylabel('Tempo de Execucao (s)')
        ax2.set_title('Escalabilidade Temporal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Tempo vs geracoes
        ax3 = axes[1, 0]
        for n in df['n_caixas'].unique():
            subset = df[df['n_caixas'] == n]
            if len(subset) > 1:
                ax3.plot(subset['geracoes_executadas'], subset['tempo'], 'o-', label=f'{n} caixas')
        ax3.set_xlabel('Geracoes Executadas')
        ax3.set_ylabel('Tempo (s)')
        ax3.set_title('Tempo vs Geracoes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Taxa de sucesso
        ax4 = axes[1, 1]
        df_sucesso = df.groupby('n_caixas').agg({'sucesso': 'mean'}).reset_index()
        ax4.bar(df_sucesso['n_caixas'], df_sucesso['sucesso'] * 100, color='green', alpha=0.7)
        ax4.set_xlabel('Numero de Caixas')
        ax4.set_ylabel('Taxa de Sucesso (%)')
        ax4.set_title('Viabilidade por Tamanho do Problema')
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        fig_file = f"{self.pasta_testes}/escalabilidade_analise.png"
        plt.savefig(fig_file, dpi=150)
        print(f"\nGrafico salvo: {fig_file}")
        plt.show()
        
        # Relatorio texto
        with open(f"{self.pasta_testes}/relatorio_analise.txt", 'w', encoding='utf-8') as f:
            f.write("RELATORIO DE ANALISE DE ESCALABILIDADE\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total de testes: {len(df)}\n")
            f.write(f"Testes bem-sucedidos: {df['sucesso'].sum()}\n\n")
            
            # Recomendacoes
            f.write("RECOMENDACOES:\n")
            f.write("-"*30 + "\n")
            
            df_sucesso_100 = df[df['sucesso']]
            if not df_sucesso_100.empty:
                max_caixas = df_sucesso_100['n_caixas'].max()
                f.write(f"1. Limite pratico de caixas: ~{max_caixas} caixas\n")
            
            # Tempo medio por caixa
            df_tempo = df[df['tempo'] > 0]
            if not df_tempo.empty:
                tempo_medio = df_tempo['tempo'].mean()
                f.write(f"2. Tempo medio de execucao: {tempo_medio:.2f}s\n")
            
            f.write("3. Recomenda-se usar max_geracoes = 200-500 para problemas medios\n")
            f.write("4. Caixas heterogeneas sao mais dificeis que homogeneas\n")
        
        print(f"\nRelatorio salvo em: {self.pasta_testes}/relatorio_analise.txt")
        
        # Salvar dados em CSV
        df.to_csv(f"{self.pasta_testes}/dados_completos.csv", index=False)
        print(f"Dados salvos em: {self.pasta_testes}/dados_completos.csv")


def main():
    print("\n" + "="*60)
    print("ANALISE DE ESCALABILIDADE DO BRKGA 3D")
    print("="*60)
    
    analisador = AnaliseEscalabilidade()
    
    # Executar estudos
    analisador.estudar_convergencia()
    analisador.estudar_tempo_por_caixa()
    analisador.estudar_container_size()
    analisador.comparar_homogeneo_heterogeneo()
    
    # Gerar relatorio
    analisador.gerar_relatorio()
    
    print("\n" + "="*60)
    print("ANALISE CONCLUIDA")
    print("="*60)


if __name__ == "__main__":
    main()