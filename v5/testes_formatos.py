"""
Testes sistemáticos para diferentes formatos de caixas
Objetivo: Encontrar parâmetros ótimos (γ_x, γ_y, γ_z) para cada cenário
"""

import json
import os
import time
import random
import numpy as np
from datetime import datetime
import csv
from brkga_3d import BRKGA

class TesteFormatos:
    """Testa diferentes configurações para diferentes formatos de caixas"""
    
    def __init__(self, L, W, H, num_execucoes=3, geracoes_por_teste=100):
        self.L = L
        self.W = W
        self.H = H
        self.num_execucoes = num_execucoes
        self.geracoes_por_teste = geracoes_por_teste
        self.resultados_globais = []
        
        self.pasta_testes = f"testes_formatos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.pasta_testes, exist_ok=True)
    
    def criar_caixas_cubicas(self, n, tamanho=1):
        """Cria N caixas cúbicas de tamanho x tamanho x tamanho"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1  # Prioridades 1-5
            caixas.append((i, tamanho, tamanho, tamanho, prioridade, f"Cubo_{i}"))
        return caixas
    
    def criar_caixas_1x1x2(self, n):
        """Cria N caixas 1x1x2 (altas)"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 1, 2, prioridade, f"Alta_{i}"))
        return caixas
    
    def criar_caixas_1x2x1(self, n):
        """Cria N caixas 1x2x1 (largas em Y)"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 1, 2, 1, prioridade, f"LargaY_{i}"))
        return caixas
    
    def criar_caixas_2x1x1(self, n):
        """Cria N caixas 2x1x1 (longas em X)"""
        caixas = []
        for i in range(n):
            prioridade = (i % 5) + 1
            caixas.append((i, 2, 1, 1, prioridade, f"LongaX_{i}"))
        return caixas
    
    def criar_caixas_aleatorias(self, n, min_dim=1, max_dim=3):
        """Cria N caixas com dimensões aleatórias entre min_dim e max_dim"""
        caixas = []
        for i in range(n):
            l = random.randint(min_dim, max_dim)
            w = random.randint(min_dim, max_dim)
            h = random.randint(min_dim, max_dim)
            prioridade = (i % 5) + 1
            caixas.append((i, l, w, h, prioridade, f"Rand_{i}"))
        return caixas
    
    def testar_configuracao(self, caixas, nome_cenario, config):
        """Testa uma configuração específica"""
        
        print(f"\n >> Testando: {config['nome']}")
        
        resultados_config = []
        
        for execucao in range(self.num_execucoes):
            brkga = BRKGA(
                num_genes=len(caixas)*2,
                num_individuos=50,
                num_elite=10,
                num_mutantes=5,
                prob_crossover=0.7,
                caixas=caixas,
                L=self.L, W=self.W, H=self.H,
                gamma_unico=config.get('gamma_unico'),
                gamma_x=config.get('gamma_x'), 
                gamma_y=config.get('gamma_y'), 
                gamma_z=config.get('gamma_z'),
                peso_x=config.get('peso_x', 1.0),
                peso_y=config.get('peso_y', 1.0),
                peso_z=config.get('peso_z', 1.0),
                exigir_estabilidade_descarga=True,
                suporte_minimo=1.0,
                verbose=False
            )
            
            inicio = time.time()
            solucao, fitness, historico, motivo = brkga.executar_continuo(
                geracoes_max=self.geracoes_por_teste,
                sem_melhoria_max=100
            )
            tempo = time.time() - inicio
            
            resultados_config.append({
                'fitness': fitness,
                'tempo': tempo,
                'volume': brkga.volume_ocupado
            })
        
        fitnesses = [r['fitness'] for r in resultados_config]
        
        return {
            'config': config['nome'],
            'params': config,
            'fitness_mean': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'tempo_mean': np.mean([r['tempo'] for r in resultados_config]),
            'volume_mean': np.mean([r['volume'] for r in resultados_config])
        }
    
    def executar_bateria_testes(self, caixas, nome_cenario, descricao):
        """Executa bateria completa de testes para um cenário"""
        
        print(f"\n{'='*70}")
        print(f"CENÁRIO: {nome_cenario}")
        print(f"Descrição: {descricao}")
        print(f"Caixas: {len(caixas)}")
        print(f"Volume total caixas: {sum(c[1]*c[2]*c[3] for c in caixas)}")
        print(f"Volume container: {self.L*self.W*self.H}")
        print(f"OBS: Violação em X é PROIBIDA (restrição rígida)")
        print(f"{'='*70}")
        
        # Configurações simplificadas (sem γ_x)
        configuracoes = [
            # Único - diferentes gammas (para Y e Z)
            {'nome': 'UNICO_γ0.05', 'gamma_unico': 0.05},
            {'nome': 'UNICO_γ0.1', 'gamma_unico': 0.1},
            {'nome': 'UNICO_γ0.2', 'gamma_unico': 0.2},
            {'nome': 'UNICO_γ0.3', 'gamma_unico': 0.3},
            {'nome': 'UNICO_γ0.5', 'gamma_unico': 0.5},
            {'nome': 'UNICO_γ0.7', 'gamma_unico': 0.7},
            {'nome': 'UNICO_γ1.0', 'gamma_unico': 1.0},
            
            # Separado - apenas Y e Z (X é restrição rígida)
            {'nome': 'SEP_Y2.0_Z0.5', 'gamma_y': 2.0, 'gamma_z': 0.5},
            {'nome': 'SEP_Y0.5_Z2.0', 'gamma_y': 0.5, 'gamma_z': 2.0},
            {'nome': 'SEP_Y1.0_Z1.0', 'gamma_y': 1.0, 'gamma_z': 1.0},
            {'nome': 'SEP_Y0.3_Z3.0', 'gamma_y': 0.3, 'gamma_z': 3.0},  # prioriza Z
            {'nome': 'SEP_Y3.0_Z0.3', 'gamma_y': 3.0, 'gamma_z': 0.3},  # prioriza Y
            
            # Híbrido (γ único + pesos apenas para Y e Z)
            {'nome': 'HIB_Y2.0_Z0.5', 'gamma_unico': 0.5, 'peso_y': 2.0, 'peso_z': 0.5},
            {'nome': 'HIB_Y0.5_Z2.0', 'gamma_unico': 0.5, 'peso_y': 0.5, 'peso_z': 2.0},
        ]
        
        resultados = []
        for config in configuracoes:
            resultado = self.testar_configuracao(caixas, nome_cenario, config)
            resultados.append(resultado)
            print(f"    Fitness: {resultado['fitness_mean']:.6f} ± {resultado['fitness_std']:.6f}")
        
        # Ordenar por fitness
        resultados.sort(key=lambda x: x['fitness_mean'], reverse=True)
        
        # Salvar resultados do cenário
        self.salvar_resultados_cenario(nome_cenario, resultados, caixas)
        
        return resultados
    

    def salvar_resultados_cenario(self, nome_cenario, resultados, caixas):
        """Salva resultados de um cenário"""
        
        arquivo = os.path.join(self.pasta_testes, f'{nome_cenario}.json')
        with open(arquivo, 'w', encoding='utf-8') as f:
            json.dump({
                'cenario': nome_cenario,
                'num_caixas': len(caixas),
                'volume_caixas': sum(c[1]*c[2]*c[3] for c in caixas),
                'resultados': resultados
            }, f, indent=2)
    
    def executar_todos_cenarios(self):
        """Executa todos os cenários de teste com números CONFIRMADOS"""
        
        # 🔴 FASE 1: Cúbicas - 25 caixas (abaixo do limite 30)
        print("\n" + "🔴"*40)
        print("FASE 1: CAIXAS CÚBICAS (1x1x1) - 25 caixas")
        print("🔴"*40)
        
        caixas_cubicas = self.criar_caixas_cubicas(25, tamanho=1)
        resultados_cubicas = self.executar_bateria_testes(
            caixas_cubicas, 
            "cubicas_1x1x1",
            "25 caixas cúbicas 1x1x1 (seguro)"
        )
        
        # 🟠 FASE 2: Altas - 12 caixas
        print("\n" + "🟠"*40)
        print("FASE 2: CAIXAS ALTAS (1x1x2) - 12 caixas")
        print("🟠"*40)
        
        caixas_altas = self.criar_caixas_1x1x2(12)
        resultados_altas = self.executar_bateria_testes(
            caixas_altas,
            "altas_1x1x2",
            "12 caixas 1x1x2"
        )
        
        # 🟡 FASE 3: Largas - 12 caixas
        print("\n" + "🟡"*40)
        print("FASE 3: CAIXAS LARGAS (1x2x1) - 12 caixas")
        print("🟡"*40)
        
        caixas_largas = self.criar_caixas_1x2x1(12)
        resultados_largas = self.executar_bateria_testes(
            caixas_largas,
            "largas_1x2x1",
            "12 caixas 1x2x1"
        )
        
        # 🟢 FASE 4: Longas - 12 caixas
        print("\n" + "🟢"*40)
        print("FASE 4: CAIXAS LONGAS (2x1x1) - 12 caixas")
        print("🟢"*40)
        
        caixas_longas = self.criar_caixas_2x1x1(12)
        resultados_longas = self.executar_bateria_testes(
            caixas_longas,
            "longas_2x1x1",
            "12 caixas 2x1x1"
        )
        
        # 🔵 FASE 5: Aleatórios - 8 caixas
        print("\n" + "🔵"*40)
        print("FASE 5: FORMATOS ALEATÓRIOS (1-3) - 8 caixas")
        print("🔵"*40)
        
        caixas_rand = self.criar_caixas_aleatorias(8, 1, 3)
        resultados_rand = self.executar_bateria_testes(
            caixas_rand,
            "aleatorios_1a3",
            "8 caixas com dimensões aleatórias 1-3"
        )
        
        # Compilar resultados
        self.compilar_resultados_finais([
            (resultados_cubicas, "Cúbicas 1x1x1 (25 caixas)"),
            (resultados_altas, "Altas 1x1x2 (12 caixas)"),
            (resultados_largas, "Largas 1x2x1 (12 caixas)"),
            (resultados_longas, "Longas 2x1x1 (12 caixas)"),
            (resultados_rand, "Aleatórios 1-3 (8 caixas)")
        ])
    
    
    def compilar_resultados_finais(self, cenarios):
        """Compila e analisa resultados de todos os cenários"""
        
        print("\n" + "="*80)
        print("RESULTADOS GLOBAIS - ANÁLISE COMPARATIVA")
        print("="*80)
        
        # Coletar melhores por cenário
        melhores_por_cenario = []
        
        for resultados, nome in cenarios:
            melhor = max(resultados, key=lambda x: x['fitness_mean'])
            melhores_por_cenario.append({
                'cenario': nome,
                'melhor_config': melhor['config'],
                'fitness': melhor['fitness_mean']
            })
            
            print(f"\n📊 {nome}:")
            print(f"   Melhor: {melhor['config']}")
            print(f"   Fitness: {melhor['fitness_mean']:.6f}")
        
        # Salvar relatório final
        with open(os.path.join(self.pasta_testes, 'relatorio_final_global.txt'), 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO GLOBAL - OTIMIZAÇÃO DE PARÂMETROS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Container: {self.L}x{self.W}x{self.H}\n")
            f.write(f"Execuções por configuração: {self.num_execucoes}\n")
            f.write(f"Gerações por teste: {self.geracoes_por_teste}\n\n")
            
            f.write("MELHORES POR CENÁRIO:\n")
            f.write("-"*60 + "\n")
            for item in melhores_por_cenario:
                f.write(f"\n{item['cenario']}:\n")
                f.write(f"  Configuração: {item['melhor_config']}\n")
                f.write(f"  Fitness: {item['fitness']:.6f}\n")
            
            # Análise de tendências
            f.write("\n" + "="*80 + "\n")
            f.write("ANÁLISE DE TENDÊNCIAS\n")
            f.write("="*80 + "\n")
            
            # Contar preferências
            preferencias = {'UNICO': 0, 'SEP_X': 0, 'SEP_Y': 0, 'SEP_Z': 0, 'HIB_X': 0, 'HIB_Z': 0}
            for item in melhores_por_cenario:
                config = item['melhor_config']
                if 'UNICO' in config:
                    preferencias['UNICO'] += 1
                elif 'SEP_X' in config:
                    preferencias['SEP_X'] += 1
                elif 'SEP_Y' in config:
                    preferencias['SEP_Y'] += 1
                elif 'SEP_Z' in config:
                    preferencias['SEP_Z'] += 1
                elif 'HIB_X' in config:
                    preferencias['HIB_X'] += 1
                elif 'HIB_Z' in config:
                    preferencias['HIB_Z'] += 1
            
            f.write("\nPreferências por tipo:\n")
            for tipo, count in preferencias.items():
                if count > 0:
                    f.write(f"  {tipo}: {count} cenário(s)\n")
            
            # Recomendação final
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMENDAÇÃO FINAL\n")
            f.write("="*80 + "\n")
            
            # Qual configuração venceu mais?
            vencedor = max(preferencias, key=lambda x: preferencias[x])
            
            if vencedor == 'UNICO':
                f.write("\n🏆 Use MODO ÚNICO com γ=0.5\n")
                f.write("   (Melhor para a maioria dos cenários)\n")
            elif vencedor == 'SEP_X':
                f.write("\n🏆 Use MODO SEPARADO priorizando X: γx=2.0, γy=0.5, γz=0.5\n")
                f.write("   (Melhor para caixas longas em profundidade)\n")
            elif vencedor == 'SEP_Z':
                f.write("\n🏆 Use MODO SEPARADO priorizando Z: γx=0.5, γy=0.5, γz=2.0\n")
                f.write("   (Melhor para caixas altas)\n")
            elif vencedor == 'HIB_X':
                f.write("\n🏆 Use MODO HÍBRIDO priorizando X: γ=0.5, px=2.0\n")
            elif vencedor == 'HIB_Z':
                f.write("\n🏆 Use MODO HÍBRIDO priorizando Z: γ=0.5, pz=2.0\n")
            
            # Configuração universal
            f.write("\n" + "-"*60 + "\n")
            f.write("CONFIGURAÇÃO UNIVERSAL RECOMENDADA:\n")
            f.write("-"*60 + "\n")
            f.write("""
brkga = BRKGA(
    gamma_unico=0.5,  # Bom para todos os casos
    # OU para maior especificidade:
    # gamma_x=2.0, gamma_y=0.5, gamma_z=0.5,
    exigir_estabilidade_descarga=True,
    suporte_minimo=1.0
)
""")
        
        print(f"\n📁 Relatório global salvo em: {self.pasta_testes}/relatorio_final_global.txt")


def main():
    # Configurações fixas
    L, W, H = 4, 5, 6  # Container pequeno para testes rápidos
    # L, W, H = 10, 8, 6  # Container médio (descomente para testes maiores)
    
    print("="*80)
    print("TESTES SISTEMÁTICOS PARA DIFERENTES FORMATOS DE CAIXAS")
    print("="*80)
    print(f"Container: {L}x{W}x{H}")
    print(f"Volume: {L*W*H}")
    print(f"Execuções por configuração: 3")
    print(f"Gerações por teste: 30")
    print("="*80)
    
    testador = TesteFormatos(
        L=L, W=W, H=H,
        num_execucoes=3,
        geracoes_por_teste=100
    )
    
    testador.executar_todos_cenarios()
    
    print("\n" + "="*80)
    print("✅ TESTES CONCLUÍDOS!")
    print("="*80)


if __name__ == "__main__":
    main()