import pygame
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from pygame.locals import *
import numpy as np
# Constantes de cores
COLORS = [
    (255, 50, 50),    # Vermelho
    (50, 255, 50),    # Verde
    (50, 50, 255),    # Azul
    (255, 255, 50),   # Amarelo
    (255, 50, 255),   # Magenta
    (50, 255, 255),   # Ciano
    (255, 150, 50),   # Laranja
    (150, 50, 255),   # Roxo
    (50, 150, 255),   # Azul claro
    (255, 100, 150),  # Rosa
    (100, 255, 150),  # Verde claro
    (150, 255, 100),  # Verde limão
]

@dataclass
class Box:
    id: int
    w: int
    h: int
    d: int
    weight: float
    max_stack_weight: float
    retrieval_priority: int

@dataclass
class Placement:
    box: Box
    x: int
    y: int
    z: int
    rotated: Tuple[int, int, int]
    color: Tuple[int, int, int]

class Container:
    def __init__(self, W: int, H: int, D: int):
        self.W = W
        self.H = H
        self.D = D
        self.placements: List[Placement] = []

    def intersects(self, a: Placement, b: Placement) -> bool:
        ax, ay, az = a.x, a.y, a.z
        aw, ah, ad = a.rotated
        bx, by, bz = b.x, b.y, b.z
        bw, bh, bd = b.rotated

        return not (
            ax + aw <= bx or bx + bw <= ax or
            ay + ah <= by or by + bh <= ay or
            az + ad <= bz or bz + bd <= az
        )

    def inside(self, p: Placement) -> bool:
        x, y, z = p.x, p.y, p.z
        w, h, d = p.rotated
        return (x >= 0 and y >= 0 and z >= 0 and
                x + w <= self.W and y + h <= self.H and z + d <= self.D)

    def support_ok(self, p: Placement) -> bool:
        if p.z == 0:
            return True
        
        required_area = p.rotated[0] * p.rotated[1] * 0.5
        supported = 0

        for other in self.placements:
            ox, oy, oz = other.x, other.y, other.z
            ow, oh, od = other.rotated

            if oz + od != p.z:
                continue

            overlap_x = max(0, min(p.x + p.rotated[0], ox + ow) - max(p.x, ox))
            overlap_y = max(0, min(p.y + p.rotated[1], oy + oh) - max(p.y, oy))
            supported += overlap_x * overlap_y

        return supported >= required_area

    def stacking_ok(self, p: Placement) -> bool:
        for other in self.placements:
            ox, oy, oz = other.x, other.y, other.z
            ow, oh, od = other.rotated

            if oz + od == p.z:
                overlap_x = max(0, min(p.x + p.rotated[0], ox + ow) - max(p.x, ox))
                overlap_y = max(0, min(p.y + p.rotated[1], oy + oh) - max(p.y, oy))

                if overlap_x > 0 and overlap_y > 0:
                    if p.box.weight > other.box.max_stack_weight:
                        return False
        return True

    def retrieval_ok(self, p: Placement) -> bool:
        for other in self.placements:
            if p.box.retrieval_priority < other.box.retrieval_priority:
                if other.z == p.z and other.x < p.x:
                    if (p.y < other.y + other.rotated[1] and 
                        p.y + p.rotated[1] > other.y):
                        return False
                
                if other.z > p.z:
                    overlap_x = max(0, min(p.x + p.rotated[0], other.x + other.rotated[0]) - max(p.x, other.x))
                    overlap_y = max(0, min(p.y + p.rotated[1], other.y + other.rotated[1]) - max(p.y, other.y))
                    
                    if overlap_x > 0 and overlap_y > 0:
                        return False
        return True

    def valid(self, p: Placement) -> bool:
        if not self.inside(p): return False
        if not self.support_ok(p): return False
        if not self.stacking_ok(p): return False
        if not self.retrieval_ok(p): return False
        
        for other in self.placements:
            if self.intersects(p, other):
                return False
        return True

    def add(self, p: Placement):
        self.placements.append(p)

    def volume_used(self) -> int:
        return sum(p.rotated[0] * p.rotated[1] * p.rotated[2] for p in self.placements)

class Visualizer3D:
    def __init__(self, container: Container, width: int = 1024, height: int = 768):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Container Packing Visualizer")
        
        self.container = container
        self.clock = pygame.time.Clock()
        
        # Parâmetros da câmera
        self.camera_distance = 500
        self.rot_x = -90-10
        self.rot_y = -22
        self.zoom = 20.0
        self.target_x = container.W / 2
        self.target_y = container.H / 2
        self.target_z = container.D / 2
            
        self.dragging = False
        self.last_mouse = (0, 0)
        
        # Configuração de fonte
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Cores
        self.bg_color = (30, 30, 40)
        self.container_color = (100, 100, 120)
        self.highlight_color = (255, 255, 100)
        
        self.selected_box = None
        self.info_panel = True
        
    def project_3d_to_2d(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Projeta coordenadas 3D para 2D com rotação e translação"""
        # Transladar para origem
        x -= self.target_x
        y -= self.target_y
        z -= self.target_z
        
        # Rotação em X
        cos_x = math.cos(math.radians(self.rot_x))
        sin_x = math.sin(math.radians(self.rot_x))
        y1 = y * cos_x - z * sin_x
        z1 = y * sin_x + z * cos_x
        
        # Rotação em Y
        cos_y = math.cos(math.radians(self.rot_y))
        sin_y = math.sin(math.radians(self.rot_y))
        x2 = x * cos_y + z1 * sin_y
        z2 = -x * sin_y + z1 * cos_y
        
        # Projeção perspectiva
        scale = self.zoom * 300 / (self.camera_distance + z2)
        screen_x = self.width // 2 + x2 * scale
        screen_y = self.height // 2 - y1 * scale
        
        return screen_x, screen_y
    
    def draw_line_3d(self, x1, y1, z1, x2, y2, z2, color, width=1):
        """Desenha uma linha 3D"""
        p1 = self.project_3d_to_2d(x1, y1, z1)
        p2 = self.project_3d_to_2d(x2, y2, z2)
        pygame.draw.line(self.screen, color, p1, p2, width)
    
    def draw_box(self, placement: Placement, highlight: bool = False):
        """Desenha uma caixa 3D"""
        x, y, z = placement.x, placement.y, placement.z
        w, h, d = placement.rotated
        
        # Vértices da caixa
        vertices = [
            (x, y, z), (x+w, y, z), (x+w, y+h, z), (x, y+h, z),  # Fundo
            (x, y, z+d), (x+w, y, z+d), (x+w, y+h, z+d), (x, y+h, z+d)  # Topo
        ]
        
        # Arestas
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Fundo
            (4,5), (5,6), (6,7), (7,4),  # Topo
            (0,4), (1,5), (2,6), (3,7)   # Verticais
        ]
        
        # Cor com base na prioridade ou destaque
        if highlight:
            color = self.highlight_color
        else:
            # Cor base da caixa
            color = placement.color
            # Tornar mais claro baseado na altura
            brightness = 0.7 + (z / self.container.D) * 0.3
            color = tuple(int(c * brightness) for c in color)
        
        # Desenhar arestas
        for edge in edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]
            self.draw_line_3d(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], color, 2)
        
        # Desenhar faces semitransparentes
        faces = [
            [0,1,2,3],  # Fundo
            [4,5,6,7],  # Topo
            [0,1,5,4],  # Frente
            [2,3,7,6],  # Trás
            [1,2,6,5],  # Direita
            [0,3,7,4]   # Esquerda
        ]
        
        for face in faces:
            points = [self.project_3d_to_2d(vertices[i][0], vertices[i][1], vertices[i][2]) for i in face]
            if highlight:
                surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.polygon(surf, (*color, 128), points)
                self.screen.blit(surf, (0, 0))
            else:
                pygame.draw.polygon(self.screen, (*color, 50), points, 1)
    
    def draw_container(self):
        """Desenha o container"""
        # Arestas do container
        corners = [
            (0,0,0), (self.container.W,0,0), (self.container.W,self.container.H,0), (0,self.container.H,0),
            (0,0,self.container.D), (self.container.W,0,self.container.D),
            (self.container.W,self.container.H,self.container.D), (0,self.container.H,self.container.D)
        ]
        
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        for edge in edges:
            p1 = corners[edge[0]]
            p2 = corners[edge[1]]
            self.draw_line_3d(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], self.container_color, 3)
        
        # Desenhar grid no piso
        for i in range(0, self.container.W + 1, max(1, self.container.W // 10)):
            self.draw_line_3d(i, 0, 0, i, self.container.H, 0, self.container_color, 1)
        for j in range(0, self.container.H + 1, max(1, self.container.H // 10)):
            self.draw_line_3d(0, j, 0, self.container.W, j, 0, self.container_color, 1)
    
    def draw_info_panel(self):
        """Desenha o painel de informações"""
        if not self.info_panel:
            return
        
        # Fundo semitransparente
        panel_surface = pygame.Surface((300, 200))
        panel_surface.set_alpha(200)
        panel_surface.fill((0, 0, 0))
        self.screen.blit(panel_surface, (10, 10))
        
        # Informações
        y = 20
        texts = [
            f"Container: {self.container.W}x{self.container.H}x{self.container.D}",
            f"Volume usado: {self.container.volume_used():,} / {self.container.W*self.container.H*self.container.D:,}",
            f"Ocupação: {100*self.container.volume_used()/(self.container.W*self.container.H*self.container.D):.1f}%",
            f"Caixas: {len(self.container.placements)}",
            "",
            "Controles:",
            "Mouse arrastar - Rotacionar",
            "Scroll - Zoom",
            "Botão direito - Selecionar caixa",
            "I - Info panel",
            "R - Reset camera",
            "ESC - Sair"
        ]
        
        for text in texts:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (20, y))
            y += 25
        
        if self.selected_box:
            box = self.selected_box.box
            y += 10
            info_texts = [
                f"Caixa selecionada: {box.id}",
                f"Dimensões: {box.w}x{box.h}x{box.d}",
                f"Posição: ({self.selected_box.x},{self.selected_box.y},{self.selected_box.z})",
                f"Peso: {box.weight}",
                f"Prioridade: {box.retrieval_priority}"
            ]
            for text in info_texts:
                surface = self.small_font.render(text, True, (255, 255, 100))
                self.screen.blit(surface, (20, y))
                y += 20
    
    def handle_events(self):
        """Processa eventos do Pygame"""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Botão esquerdo
                    self.dragging = True
                    self.last_mouse = event.pos
                elif event.button == 3:  # Botão direito - selecionar caixa
                    self.select_box_at_mouse(event.pos)
                elif event.button == 4:  # Scroll up
                    self.zoom *= 1.1
                elif event.button == 5:  # Scroll down
                    self.zoom /= 1.1
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            elif event.type == MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse[0]
                    dy = event.pos[1] - self.last_mouse[1]
                    self.rot_y += dx * 0.5
                    self.rot_x += dy * 0.5
                    self.last_mouse = event.pos
            
            elif event.type == KEYDOWN:
                if event.key == K_i:
                    self.info_panel = not self.info_panel
                elif event.key == K_r:
                    self.reset_camera()
        
        return True
    
    def reset_camera(self):
        """Reseta a câmera para a posição inicial"""
        self.rot_x = 30
        self.rot_y = 45
        self.zoom = 1.0
        self.camera_distance = 500
    
    def select_box_at_mouse(self, mouse_pos):
        """Seleciona a caixa sob o cursor do mouse"""
        closest_distance = float('inf')
        closest_box = None
        
        for placement in reversed(self.container.placements):
            # Verificar se o mouse está sobre a caixa (detecção simplificada)
            x, y, z = placement.x, placement.y, placement.z
            w, h, d = placement.rotated
            
            # Projetar centro da caixa
            cx, cy, cz = x + w/2, y + h/2, z + d/2
            screen_pos = self.project_3d_to_2d(cx, cy, cz)
            
            distance = math.sqrt((mouse_pos[0] - screen_pos[0])**2 + (mouse_pos[1] - screen_pos[1])**2)
            if distance < closest_distance and distance < 50:
                closest_distance = distance
                closest_box = placement
        
        self.selected_box = closest_box
    
    def draw_legend(self):
        """Desenha legenda de cores por prioridade"""
        legend_x = self.width - 200
        legend_y = 20
        
        # Fundo
        panel_surface = pygame.Surface((180, 150))
        panel_surface.set_alpha(200)
        panel_surface.fill((0, 0, 0))
        self.screen.blit(panel_surface, (legend_x, legend_y))
        
        # Título
        title = self.font.render("Prioridades de Retirada", True, (255, 255, 255))
        self.screen.blit(title, (legend_x + 10, legend_y + 10))
        
        priorities = {}
        for p in self.container.placements:
            prio = p.box.retrieval_priority
            if prio not in priorities:
                priorities[prio] = p.color
        
        y = legend_y + 40
        for prio, color in sorted(priorities.items()):
            pygame.draw.rect(self.screen, color, (legend_x + 10, y, 20, 20))
            text = self.small_font.render(f"Prioridade {prio}", True, (255, 255, 255))
            self.screen.blit(text, (legend_x + 40, y + 3))
            y += 25
    
    def render(self):
        """Renderiza a cena completa"""
        self.screen.fill(self.bg_color)
        
        # Desenhar grid de fundo
        for i in range(0, 100, 10):
            self.draw_line_3d(-1000, i, -1000, 1000, i, -1000, (50, 50, 60), 1)
        
        # Desenhar container e caixas
        self.draw_container()
        
        # Ordenar caixas por distância para renderização correta
        sorted_placements = sorted(self.container.placements, 
                                 key=lambda p: -(p.x + p.y + p.z))
        
        for placement in sorted_placements:
            highlight = (self.selected_box == placement)
            self.draw_box(placement, highlight)
        
        # Desenhar interfaces
        self.draw_info_panel()
        self.draw_legend()
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def run(self):
        """Loop principal da visualização"""
        running = True
        t=0
        while running:
            t+=1
            #self.rot_y = 0   + 30 * np.sin((t+math.pi/2)/100)
            # self.rot_x = -90 + 10 * np.sin((t+math.pi/2)/100)
            running = self.handle_events()
            self.render()
        
        pygame.quit()
        sys.exit()

def rotations(box: Box) -> List[Tuple[int, int, int]]:
    """Gera todas as 6 rotações possíveis"""
    w, h, d = box.w, box.h, box.d
    return list({
        (w, h, d), (w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w),
    })

def extreme_points(container: Container) -> List[Tuple[int, int, int]]:
    """Gera pontos extremos válidos"""
    points: Set[Tuple[int, int, int]] = {(0, 0, 0)}
    
    for p in container.placements:
        x, y, z = p.x, p.y, p.z
        w, h, d = p.rotated
        
        if z + d <= container.D:
            points.add((x, y, z + d))
        if x + w <= container.W:
            points.add((x + w, y, z))
        if y + h <= container.H:
            points.add((x, y + h, z))
        if x + w <= container.W and y + h <= container.H:
            points.add((x + w, y + h, z))
        if x + w <= container.W and z + d <= container.D:
            points.add((x + w, y, z + d))
        if y + h <= container.H and z + d <= container.D:
            points.add((x, y + h, z + d))
    
    valid_points = []
    for x, y, z in points:
        if (0 <= x <= container.W and 0 <= y <= container.H and 0 <= z <= container.D):
            occupied = False
            for p in container.placements:
                if (x >= p.x and x < p.x + p.rotated[0] and
                    y >= p.y and y < p.y + p.rotated[1] and
                    z >= p.z and z < p.z + p.rotated[2]):
                    occupied = True
                    break
            if not occupied:
                valid_points.append((x, y, z))
    
    return sorted(valid_points, key=lambda t: (t[0] + t[1] + t[2]))

def pack(container: Container, boxes: List[Box], visualizer=None) -> bool:
    """Algoritmo de empacotamento com visualização opcional"""
    boxes = sorted(boxes, key=lambda b: (b.retrieval_priority, -(b.w * b.h * b.d), -b.weight))
    
    color_index = 0
    
    for box in boxes:
        placed = False
        best_placement = None
        
        pts = extreme_points(container)
        
        for rot in rotations(box):
            for x, y, z in pts:
                p = Placement(box, x, y, z, rot, COLORS[color_index % len(COLORS)])
                
                if container.valid(p):
                    if best_placement is None or p.z < best_placement.z:
                        best_placement = p
        
        if best_placement:
            container.add(best_placement)
            color_index += 1
            placed = True
            
            # Opcional: renderizar a cada caixa colocada
            if visualizer:
                visualizer.render()
                pygame.time.wait(500)  # Pausa para visualização
        
        if not placed:
            print(f"Falha ao posicionar caixa {box.id}")
            return False
    
    return True

def main():
    # Configuração do container
    container = Container(20, 15, 12)
    
    # Lista de caixas
    boxes = [
        Box(1, 8, 6, 5, 25, 200, 1),
        Box(2, 7, 5, 4, 20, 150, 1),
        Box(3, 6, 6, 6, 30, 200, 2),
        Box(4, 5, 5, 5, 15, 100, 2),
        Box(5, 4, 4, 6, 12, 80, 3),
        Box(6, 5, 3, 4, 10, 60, 3),
        Box(7, 4, 4, 4, 8, 50, 3),
        Box(8, 3, 3, 3, 5, 30, 4),
        Box(9, 2, 2, 5, 4, 25, 4),
        Box(10, 3, 2, 2, 3, 20, 4),
        Box(11, 2, 2, 2, 2, 15, 5),
        Box(12, 2, 2, 2, 2, 15, 5),
        Box(13, 2, 2, 2, 2, 10, 5),
        Box(14, 1, 8, 4, 1, 8, 5),
        Box(15, 3, 1, 2, 2, 12, 4),
        Box(16, 3, 8, 8, 2, 9, 4),
        Box(17, 1, 8, 1, 1, 5, 6),
        Box(18, 2, 8, 8, 1, 6, 6),
        Box(19, 2, 8, 8, 1, 6, 6),
        Box(20, 2, 8, 8, 1, 6, 6),
    ]
    
    print("Iniciando empacotamento...")
    print(f"Container: {container.W}x{container.H}x{container.D}")
    print(f"Caixas a empacotar: {len(boxes)}")
    
    # Criar visualizador
    visualizer = Visualizer3D(container)
    
    # Executar empacotamento
    success = pack(container, boxes, visualizer)
    
    if success:
        print("\nEmpacotamento concluído com sucesso!")
        print(f"Volume utilizado: {container.volume_used()}")
        print(f"Ocupação: {100*container.volume_used()/(container.W*container.H*container.D):.1f}%")
        print("\nIniciando visualização 3D...")
        print("Use o mouse para rotacionar a vista")
        print("Botão direito para selecionar caixas")
        print("ESC para sair")
        
        # Executar visualizador
        visualizer.run()
    else:
        print("Falha no empacotamento!")

if __name__ == "__main__":
    main()