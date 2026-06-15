from itertools import product

# Dimensões do contêiner
L, W, H = 5, 5, 3

# Dados das caixas: (l, w, h, d)
caixas = [
    (1, 1, 1, 0),  # B1: 1x1x1
    (2, 2, 1, 1),  # B2: 2x2x1
]

def gerar_posicoes(l, w, h, L, W, H):
    """Gera todas posições (x,y,z,r) possíveis para uma caixa"""
    posicoes = []
    for r in [0, 1]:
        l_ef = l if r == 0 else w
        w_ef = w if r == 0 else l
        h_ef = h
        
        if l_ef > L or w_ef > W or h_ef > H:
            continue
            
        for x in range(L - l_ef + 1):
            for y in range(W - w_ef + 1):
                for z in range(H - h_ef + 1):
                    posicoes.append((x, y, z, r))
    return posicoes

def sobrepoe(pos1, l1, w1, h1, pos2, l2, w2, h2):
    """Verifica se duas caixas se sobrepõem"""
    x1, y1, z1, r1 = pos1
    x2, y2, z2, r2 = pos2
    
    # Dimensões efetivas
    l1_ef = l1 if r1 == 0 else w1
    w1_ef = w1 if r1 == 0 else l1
    h1_ef = h1
    
    l2_ef = l2 if r2 == 0 else w2
    w2_ef = w2 if r2 == 0 else l2
    h2_ef = h2
    
    # Verifica sobreposição nos três eixos
    x_over = not (x1 + l1_ef <= x2 or x2 + l2_ef <= x1)
    y_over = not (y1 + w1_ef <= y2 or y2 + w2_ef <= y1)
    z_over = not (z1 + h1_ef <= z2 or z2 + h2_ef <= z1)
    
    return x_over and y_over and z_over

# Gerar todas posições para cada caixa
posicoes_B1 = gerar_posicoes(1, 1, 1, L, W, H)
posicoes_B2 = gerar_posicoes(2, 2, 1, L, W, H)

print(f"Posições B1: {len(posicoes_B1)}")
print(f"Posições B2: {len(posicoes_B2)}")
print(f"Combinações brutas: {len(posicoes_B1) * len(posicoes_B2)}")
print()

# Gerar todas combinações sem sobreposição
solucoes_validas = []

for pos1 in posicoes_B1:
    for pos2 in posicoes_B2:
        if not sobrepoe(pos1, 1, 1, 1, pos2, 2, 2, 1):
            solucoes_validas.append((pos1, pos2))

print(f"Combinações sem sobreposição: {len(solucoes_validas)}")
print()

# Listar todas as soluções válidas
print("=" * 60)
print("TODAS AS SOLUÇÕES POSSÍVEIS (SEM SOBREPOSIÇÃO)")
print("=" * 60)
print()

for i, (pos1, pos2) in enumerate(solucoes_validas):
    print(f"{i+1:4d}. B1={pos1}, B2={pos2}")

# Salvar em arquivo
with open("solucoes_possiveis.txt", "w") as f:
    f.write(f"Total de soluções: {len(solucoes_validas)}\n\n")
    for i, (pos1, pos2) in enumerate(solucoes_validas):
        f.write(f"{i+1}. ({pos1[0]},{pos1[1]},{pos1[2]},{pos1[3]}), ({pos2[0]},{pos2[1]},{pos2[2]},{pos2[3]})\n")

print(f"\nArquivo 'solucoes_possiveis.txt' salvo com {len(solucoes_validas)} soluções.")