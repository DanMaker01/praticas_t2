"""
Teste de Limitações do BRKGA 3D
Estuda o comportamento do algoritmo em diferentes cenários
"""

import random
import numpy as np
import json
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brkga_3d import BRKGA, converter_para_caixas

class TesteLimitacao:
    """Estuda as limitações do algoritmo em diferentes configurações"""
    
    def __init__(self):
        self.resultados = []
        self.pasta_testes = f"testes_limitacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
        
        # Configuração padrao do BRKGA (otimizada)
        self.num_individuos = 50
        self.num_elite = 10
        self.num_mutantes = 5
        self.prob_crossover = 0.7
        self.geracoes_max = 200
        self.sem_melhoria_max = 50
        self.gamma_z = 0.01
    
    def gerar_caixas_aleatorias(self, n, dim_max=3):
        """Gera n caixas com dimensoes aleatorias"""
        caixas = []
        for i in range(n):
            l = random.randint(1, dim_max)
            w = random.randint(1, dim_max)
            h = random.randint(1, dim_max)
            prioridade = (i % 5) + 1
            nome = f"Rand_{i}"
            caixas.append((i, l, w, h, prioridade, nome))
        return caixas
    
    def gerar_caixas_cubicas(self, n):
        """Gera n caixas cubicas 1x1x1"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 1, 1, prioridade, f"Cubo_{i}"))
        return caixas
    
    def gerar_caixas_mistas(self, pedidos_order):
        """Gera caixas a partir de um dicionario de pedidos"""
        return converter_para_caixas(pedidos_order)
    
    def executar_teste(self, nome, caixas, L, W, H, tempo_max_minutos=5):
        """Executa um teste e retorna os resultados"""
        
        print(f"\n{'='*60}")
        print(f"Teste: {nome}")
        print(f"Container: {L}x{W}x{H} = {L*W*H}")
        print(f"Caixas: {len(caixas)}")
        volume_caixas = sum(c[1]*c[2]*c[3] for c in caixas)
        print(f"Volume total: {volume_caixas}/{L*W*H} = {volume_caixas/(L*W*H)*100:.1f}%")
        print(f"Tempo maximo: {tempo_max_minutos} min")
        print(f"{'='*60}")
        
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
            geracoes_max=self.geracoes_max,
            sem_melhoria_max=self.sem_melhoria_max,
            tempo_maximo_horas=tempo_max_minutos/60
        )
        tempo_total = time.time() - inicio
        
        resultado = {
            'teste': nome,
            'L': L, 'W': W, 'H': H,
            'num_caixas': len(caixas),
            'volume_caixas': volume_caixas,
            'ocupacao_teorica': volume_caixas/(L*W*H)*100,
            'volume_ocupado': brkga.volume_ocupado,
            'utilizacao_real': brkga.volume_ocupado/(L*W*H)*100,
            'fitness': fitness,
            'tempo': tempo_total,
            'geracoes': brkga.geracao_atual,
            'solucoes_invalidas': brkga.solucoes_invalidas,
            'motivo': motivo,
            'sucesso': fitness != -float('inf')
        }
        
        self.resultados.append(resultado)
        return resultado
    
    def testar_diferentes_cargas(self):
        """Testa o algoritmo com diferentes numeros de caixas"""
        
        print("\n" + "="*50)
        print("1. Teste de Carga: Diferentes quantidades de caixas")
        print("="*50)
        
        L, W, H = 10, 8, 6
        resultados_carga = []
        
        for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            caixas = self.gerar_caixas_cubicas(n)
            resultado = self.executar_teste(f"Carga_{n}", caixas, L, W, H, tempo_max_minutos=3)
            resultados_carga.append(resultado)
            
            if not resultado['sucesso']:
                print(f"  Falha em {n} caixas")
                break
        
        return resultados_carga
    
    def testar_diferentes_tamanhos_container(self):
        """Testa o algoritmo com diferentes tamanhos de container"""
        
        print("\n" + "="*50)
        print("2. Teste de Escala: Diferentes tamanhos de container")
        print("="*50)
        
        containers = [
            (4, 5, 6, "Pequeno"),
            (6, 8, 10, "Medio"),
            (8, 10, 12, "Grande"),
            (10, 12, 15, "Extra"),
        ]
        
        n_caixas = 50
        resultados_container = []
        
        for L, W, H, nome in containers:
            caixas = self.gerar_caixas_cubicas(n_caixas)
            resultado = self.executar_teste(f"Container_{nome}", caixas, L, W, H, tempo_max_minutos=5)
            resultados_container.append(resultado)
        
        return resultados_container
    
    def testar_diferentes_volumes(self):
        """Testa o algoritmo com diferentes volumes de ocupacao"""
        
        print("\n" + "="*50)
        print("3. Teste de Volume: Diferentes ocupacoes do container")
        print("="*50)
        
        L, W, H = 8, 10, 12
        volume_total = L * W * H
        resultados_volume = []
        
        ocupacoes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for ocup in ocupacoes:
            n = int(volume_total * ocup / 8)  # caixas de volume ~8
            n = max(5, min(n, 200))
            caixas = self.gerar_caixas_aleatorias(n, dim_max=4)
            resultado = self.executar_teste(f"Volume_{int(ocup*100)}", caixas, L, W, H, tempo_max_minutos=5)
            resultados_volume.append(resultado)
        
        return resultados_volume
    
    def testar_diferentes_formatos(self):
        """Testa o algoritmo com diferentes formatos de caixa"""
        
        print("\n" + "="*50)
        print("4. Teste de Formato: Diferentes proporcoes de caixas")
        print("="*50)
        
        L, W, H = 10, 8, 12
        n = 60
        resultados_formato = []
        
        formatos = [
            ("Cubicas", [(1,1,1)]),
            ("Altas", [(1,1,3)]),
            ("Largas", [(1,3,1)]),
            ("Longas", [(3,1,1)]),
            ("Mistas", [(1,1,1), (1,2,2), (2,2,1)]),
        ]
        
        for nome, dims in formatos:
            caixas = []
            for i in range(n):
                dim = random.choice(dims)
                l, w, h = dim
                prioridade = (i % 5) + 1
                caixas.append((i, l, w, h, prioridade, f"{nome}_{i}"))
            
            resultado = self.executar_teste(f"Formato_{nome}", caixas, L, W, H, tempo_max_minutos=5)
            resultados_formato.append(resultado)
        
        return resultados_formato
    
    def testar_prioridades_extremas(self):
        """Testa o algoritmo com diferentes distribuicoes de prioridade"""
        
        print("\n" + "="*50)
        print("5. Teste de Prioridade: Diferentes distribuicoes")
        print("="*50)
        
        L, W, H = 8, 10, 12
        n = 50
        resultados_prioridade = []
        
        distribuicoes = [
            ("Equilibrada", lambda i: (i % 3) + 1),
            ("Muitas_P1", lambda i: 1 if i < 30 else (i % 3) + 2),
            ("Muitas_P5", lambda i: 5 if i < 30 else (i % 3) + 1),
            ("Decrescente", lambda i: i % 5 + 1),
            ("Crescente", lambda i: 5 - (i % 5)),
        ]
        
        for nome, func in distribuicoes:
            caixas = []
            for i in range(n):
                l = random.randint(1, 3)
                w = random.randint(1, 3)
                h = random.randint(1, 3)
                prioridade = func(i)
                caixas.append((i, l, w, h, prioridade, f"{nome}_{i}"))
            
            resultado = self.executar_teste(f"Prioridade_{nome}", caixas, L, W, H, tempo_max_minutos=5)
            resultados_prioridade.append(resultado)
        
        return resultados_prioridade
    
    def testar_pedido_real(self):
        """Testa com o pedido real do usuario"""
        
        print("\n" + "="*50)
        print("6. Teste Real: Pedido do usuario")
        print("="*50)
        
        pedidos = {
            0: {(1, 1, 1): 4},
            1: {(1, 1, 4): 2},
            2: {(4, 1, 1): 2},
            3: {(3, 3, 3): 1},
            4: {(1, 1, 1): 10},
            5: {(2, 2, 1): 3, (1, 2, 2): 1}
        }
        
        caixas = converter_para_caixas(pedidos)
        L, W, H = 4, 5, 6
        
        resultado = self.executar_teste("Pedido_Real", caixas, L, W, H, tempo_max_minutos=10)
        
        return resultado
    
    def analisar_limites(self):
        """Analisa os limites do algoritmo baseado nos testes"""
        
        df = pd.DataFrame(self.resultados)
        
        print("\n" + "="*60)
        print("ANALISE DOS LIMITES DO ALGORITMO")
        print("="*60)
        
        # Taxa de sucesso geral
        sucessos = df['sucesso'].sum()
        total = len(df)
        print(f"\nTaxa de sucesso geral: {sucessos}/{total} ({sucessos/total*100:.1f}%)")
        
        # Limite de caixas
        df_carga = df[df['teste'].str.startswith('Carga_')]
        if not df_carga.empty:
            ultimo_sucesso = df_carga[df_carga['sucesso']].iloc[-1] if len(df_carga[df_carga['sucesso']]) > 0 else None
            if ultimo_sucesso is not None:
                print(f"\nLimite de caixas cubicas: ~{ultimo_sucesso['num_caixas']} caixas")
        
        # Limite de ocupacao
        df_volume = df[df['teste'].str.startswith('Volume_')]
        if not df_volume.empty:
            df_volume_sucesso = df_volume[df_volume['sucesso']]
            if not df_volume_sucesso.empty:
                max_ocup = df_volume_sucesso['ocupacao_teorica'].max()
                print(f"Limite de ocupacao: ~{max_ocup:.1f}%")
        
        # Efeito do formato
        df_formato = df[df['teste'].str.startswith('Formato_')]
        if not df_formato.empty:
            print("\nEfeito do formato das caixas:")
            for _, r in df_formato.iterrows():
                status = "OK" if r['sucesso'] else "FALHA"
                print(f"  {r['teste'].replace('Formato_','')}: {status} (ocupacao {r['ocupacao_teorica']:.1f}%)")
        
        return df
    
    def gerar_relatorio(self):
        """Gera relatorio grafico dos resultados"""
        
        df = pd.DataFrame(self.resultados)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sucesso por numero de caixas
        ax1 = axes[0, 0]
        df_carga = df[df['teste'].str.startswith('Carga_')]
        if not df_carga.empty:
            ax1.plot(df_carga['num_caixas'], df_carga['sucesso'], 'bo-', linewidth=2)
            ax1.set_xlabel('Numero de Caixas')
            ax1.set_ylabel('Sucesso (1=Sim, 0=Nao)')
            ax1.set_title('Limite de Capacidade')
            ax1.grid(True, alpha=0.3)
        
        # 2. Tempo vs caixas
        ax2 = axes[0, 1]
        if not df_carga.empty:
            ax2.plot(df_carga['num_caixas'], df_carga['tempo'], 'ro-', linewidth=2)
            ax2.set_xlabel('Numero de Caixas')
            ax2.set_ylabel('Tempo (s)')
            ax2.set_title('Escalabilidade Temporal')
            ax2.grid(True, alpha=0.3)
        
        # 3. Utilizacao vs ocupacao teorica
        ax3 = axes[1, 0]
        ax3.scatter(df['ocupacao_teorica'], df['utilizacao_real'], c=df['sucesso'].map({True: 'green', False: 'red'}), s=50)
        ax3.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Ideal')
        ax3.set_xlabel('Ocupacao Teorica (%)')
        ax3.set_ylabel('Utilizacao Real (%)')
        ax3.set_title('Eficiencia de Empacotamento')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Fitness vs ocupacao
        ax4 = axes[1, 1]
        df_sucesso = df[df['sucesso']]
        if not df_sucesso.empty:
            ax4.scatter(df_sucesso['ocupacao_teorica'], df_sucesso['fitness'], c='blue', s=50)
            ax4.set_xlabel('Ocupacao Teorica (%)')
            ax4.set_ylabel('Fitness')
            ax4.set_title('Qualidade vs Ocupacao')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fig_file = f"{self.pasta_testes}/limitacao_graficos.png"
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        print(f"\nGraficos salvos em: {fig_file}")
        plt.show()
        
        return fig
    
    def salvar_resultados(self):
        """Salva todos os resultados"""
        
        df = pd.DataFrame(self.resultados)
        
        csv_file = f"{self.pasta_testes}/resultados_completos.csv"
        df.to_csv(csv_file, index=False)
        
        json_file = f"{self.pasta_testes}/resultados_completos.json"
        with open(json_file, 'w') as f:
            json.dump(self.resultados, f, indent=2)
        
        # Relatorio de limitacoes
        with open(f"{self.pasta_testes}/limitacoes.txt", 'w') as f:
            f.write("RELATORIO DE LIMITACOES DO ALGORITMO\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Total de testes: {len(df)}\n")
            f.write(f"Testes bem sucedidos: {df['sucesso'].sum()}\n")
            f.write(f"Taxa de sucesso: {df['sucesso'].mean()*100:.1f}%\n\n")
            
            f.write("PRINCIPAIS LIMITACOES IDENTIFICADAS:\n")
            f.write("-"*40 + "\n")
            f.write("1. BLF (Bottom-Left-Front) fragmenta o espaco facilmente\n")
            f.write("2. Para muitas caixas, o algoritmo nao encontra espaco\n")
            f.write("3. Caixas grandes dificultam o empacotamento\n")
            f.write("4. A ordenacao aleatoria inicial pode ser muito ruim\n")
            f.write("5. Sem busca local, solucoes nao sao refinadas\n\n")
            
            f.write("RECOMENDACOES:\n")
            f.write("-"*40 + "\n")
            f.write("1. Implementar busca local apos o BLF\n")
            f.write("2. Usar caixas menores sempre que possivel\n")
            f.write("3. Aumentar populacao e geracoes para problemas grandes\n")
            f.write("4. Considerar MAXRECT em vez de BLF\n")
        
        print(f"\nResultados salvos em: {self.pasta_testes}/")


def main():
    print("\n" + "="*60)
    print("TESTE DE LIMITACOES DO BRKGA 3D")
    print("="*60)
    
    testador = TesteLimitacao()
    
    # Executar todos os testes
    testador.testar_diferentes_cargas()
    testador.testar_diferentes_tamanhos_container()
    testador.testar_diferentes_volumes()
    testador.testar_diferentes_formatos()
    testador.testar_prioridades_extremas()
    testador.testar_pedido_real()
    
    # Analisar e salvar
    testador.analisar_limites()
    testador.gerar_relatorio()
    testador.salvar_resultados()
    
    print("\n" + "="*60)
    print("TESTES CONCLUIDOS")
    print("="*60)


if __name__ == "__main__":
    main()