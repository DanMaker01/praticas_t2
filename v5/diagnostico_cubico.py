# diagnostico_cubico.py
"""
Descobre o número máximo de caixas 1x1x1 que podem ser posicionadas
"""

from brkga_3d import BRKGA
import random

def testar_numero_caixas(L, W, H, n_caixas):
    """Testa se um número de caixas é viável"""
    
    # Criar caixas
    caixas = []
    for i in range(n_caixas):
        prioridade = (i % 5) + 1
        caixas.append((i, 1, 1, 1, prioridade, f"Cubo_{i}"))
    
    print(f"\nTestando {n_caixas} caixas (volume: {n_caixas}/{L*W*H} = {n_caixas/(L*W*H)*100:.1f}%)")
    
    # Tentar com γ bem pequeno (mais flexível)
    brkga = BRKGA(
        num_genes=len(caixas)*2,
        num_individuos=20,  # População menor para teste rápido
        num_elite=4,
        num_mutantes=2,
        prob_crossover=0.7,
        caixas=caixas,
        L=L, W=W, H=H,
        gamma_unico=0.01,  # γ bem pequeno para aceitar mais soluções
        exigir_estabilidade_descarga=True,
        suporte_minimo=1.0,
        verbose=False
    )
    
    solucao, fitness, _, motivo = brkga.executar_continuo(
        geracoes_max=500,  # Poucas gerações para teste rápido
        sem_melhoria_max=500
    )
    
    if fitness != -float('inf'):
        print(f"  ✅ VIÁVEL! Fitness: {fitness:.6f}")
        print(f"  📦 Volume usado: {brkga.volume_ocupado}")
        return True
    else:
        print(f"  ❌ INVIÁVEL")
        return False

def main():
    L, W, H = 4, 5, 6
    volume = L * W * H
    
    print("="*60)
    print("DIAGNÓSTICO: CAPACIDADE REAL PARA CAIXAS 1x1x1")
    print("="*60)
    print(f"Container: {L}x{W}x{H} = {volume}")
    print("="*60)
    
    # Testar números crescentes
    for n in [10, 20, 30, 40, 50, 60, 70, 80]:
        if not testar_numero_caixas(L, W, H, n):
            print(f"\n🔴 LIMITE MÁXIMO: {n-10} caixas")
            print(f"   (Acima disso, o algoritmo não consegue posicionar)")
            break

if __name__ == "__main__":
    main()