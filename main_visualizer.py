import pygame
import torch
import numpy as np
import math
from typing import Dict, Any

# --- Importação dos módulos anteriores (simulada) ---
# Assumindo que as classes estão no mesmo arquivo ou importadas corretamente:
from veiculo import VehicleDynamicsGPU
from geometria_veiculo import VehicleGeometryGPU
from mapa import ParkingMapGenerator, MapEntity
from colisao_sensores import CollisionAndSensorSystemGPU
from obter_coordenadas_privilegiadas import get_privileged_coordinates
from toilered_custo import tailored_cost_function

# ==============================================================================
# 1. Configuração Física Atualizada (Compatível com VehicleDynamicsGPU)
# ==============================================================================
VEHICLE_CONFIG = {
    # Parâmetros Físicos (Para a Dinâmica)
    'length_tractor_wheelbase': 3.29,   # D
    'length_trailer_wheelbase': 7.135,  # L
    'd_axle_kingpin': -0.38,            # M (Offset da quinta roda)

    # Parâmetros Legados (Para Coordenadas Privilegiadas e Visualização Antiga)
    # Mantemos para compatibilidade com as funções de custo/geometria existentes
    'inv_D': 1.0 / 3.29,
    'inv_L': 1.0 / 7.135,
    'offset_factor': -0.38 * (1.0/7.135) * (1.0/3.29), # offset_factor = M * inv_L * inv_D

    # Geometria Visual e de Colisão
    'width_tractor': 2.63,
    'width_trailer': 2.63,
    'length_tractor': 5.5,
    'length_trailer': 13.6,
    
    # Offsets Geométricos (Visualização)
    'd_front_kingpin': 4.0,   
    'd_front_trailer_kingpin': 1.6, 
}
# ==============================================================================
# 2. Motor de Renderização (Pygame)
# ==============================================================================
class ParkingRenderer:
    def __init__(self, map_width, map_height, width=1280, height=720):
        pygame.init()
        self.screen_size = (width, height)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Truck Parking Simulator - Debug Environment")
        self.font = pygame.font.SysFont("Consolas", 14)
        
        # Câmera
        self.ppm = 10.0  # Pixels por Metro (Zoom)
        self.camera_offset = np.array([width/2, height/2])
        self.map_dims = (map_width, map_height)

    def world_to_screen(self, world_pos, focus_pos):
        """Converte coordenadas do mundo (m) para tela (px), centralizando no focus_pos."""
        # Vetor relativo ao foco (caminhão)
        rel_pos = world_pos - focus_pos
        # Inverter Y pois no Pygame Y cresce para baixo
        screen_x = self.camera_offset[0] + rel_pos[0] * self.ppm
        screen_y = self.camera_offset[1] - rel_pos[1] * self.ppm
        return (int(screen_x), int(screen_y))

    def draw_map(self, obstacles_tensor, focus_pos):
        self.screen.fill((30, 30, 30)) # Fundo Cinza Escuro
        
        # Desenhar Grade (Grid)
        grid_size = 10 # metros
        # ... lógica de grid simplificada ...

        # Desenhar Obstáculos (Paredes/Vagas)
        # Tensor: [x, y, theta, width, length]
        if obstacles_tensor is not None:
            obst_np = obstacles_tensor.cpu().numpy()
            for obs in obst_np:
                x, y, theta, w, l = obs
                # Criar os 4 cantos do retângulo rotacionado
                corners = self._get_rect_corners(x, y, theta, w, l)
                poly_points = [self.world_to_screen(p, focus_pos) for p in corners]
                
                color = (100, 100, 100) # Parede cinza
                if w > 4.0 and l > 10.0: color = (50, 80, 50) # Vaga (verde escuro)
                
                pygame.draw.polygon(self.screen, color, poly_points)

    def draw_vehicle(self, geometry_dict, focus_pos, is_colliding):
        """Desenha o trator e trailer baseados nos vértices calculados."""
        # Cores
        color_tractor = (200, 50, 50) if is_colliding else (70, 130, 180) # Azul (ou Vermelho se bater)
        color_trailer = (180, 50, 50) if is_colliding else (200, 200, 200) # Branco
        
        # Trator
        trator_verts = geometry_dict['trator'][0].cpu().numpy() # Pega o batch 0
        poly_trator = [self.world_to_screen(p, focus_pos) for p in trator_verts]
        pygame.draw.polygon(self.screen, color_tractor, poly_trator)
        pygame.draw.polygon(self.screen, (0, 0, 0), poly_trator, 2) # Contorno

        # Trailer
        trailer_verts = geometry_dict['trailer'][0].cpu().numpy()
        poly_trailer = [self.world_to_screen(p, focus_pos) for p in trailer_verts]
        pygame.draw.polygon(self.screen, color_trailer, poly_trailer)
        pygame.draw.polygon(self.screen, (0, 0, 0), poly_trailer, 2)

    def draw_debug_circles(self, centers_dict, radii, focus_pos):
        """Desenha os círculos de colisão."""
        # Trator
        t_centers = centers_dict['tractor_centers'][0].cpu().numpy()
        for c in t_centers:
            pos = self.world_to_screen(c, focus_pos)
            radius = int(radii['tractor_radius'] * self.ppm)
            pygame.draw.circle(self.screen, (0, 255, 0), pos, radius, 1) # Verde contorno
            
        # Trailer
        tr_centers = centers_dict['trailer_centers'][0].cpu().numpy()
        for c in tr_centers:
            pos = self.world_to_screen(c, focus_pos)
            radius = int(radii['trailer_radius'] * self.ppm)
            pygame.draw.circle(self.screen, (0, 255, 255), pos, radius, 1) # Ciano contorno
    
    def draw_lidar(self, lidar_data, state, focus_pos):
        """
        Desenha os raios do LiDAR na tela.
        
        Parâmetros:
            lidar_data: Tensor/Array (16,) com as distâncias.
            state: Tensor/Array (4,) [x, y, theta, beta].
            focus_pos: Ponto central da câmera.
        """
        # Converter para Numpy/CPU se necessário
        if isinstance(lidar_data, torch.Tensor):
            dists = lidar_data.detach().cpu().numpy()
        else:
            dists = lidar_data

        if isinstance(state, torch.Tensor):
            st = state.detach().cpu().numpy()
        else:
            st = state
            
        x, y, theta, beta = st
        theta_trailer = theta + beta
        
        # --- A. Recalcular Origens (Geometria) ---
        # Acessa a config global (certifique-se que VEHICLE_CONFIG está acessível)
        lt = VEHICLE_CONFIG['length_tractor']
        ltr = VEHICLE_CONFIG['length_trailer']
        d_kingpin = VEHICLE_CONFIG['d_front_trailer_kingpin']
        M = VEHICLE_CONFIG['d_axle_kingpin']
        
        # Offset Trator (Centro Geométrico aprox.)
        off_tractor = lt / 2.0
        # Offset Trailer (Centro Geométrico aprox. da carga)
        dist_pino_fundo = ltr - d_kingpin
        off_trailer = -dist_pino_fundo / 2.0

        # Centro Global do Trator
        ctx = x + math.cos(theta) * off_tractor
        cty = y + math.sin(theta) * off_tractor
        
        # Centro Global do Trailer
        # 1. Posição da Quinta Roda (Pivô)
        qx = x + math.cos(theta) * M
        qy = y + math.sin(theta) * M
        # 2. Centro a partir do Pivô
        ctrx = qx + math.cos(theta_trailer) * off_trailer
        ctry = qy + math.sin(theta_trailer) * off_trailer

        # --- B. Desenhar os 16 Raios ---
        # 8 raios por sensor, espaçados de 45 graus (pi/4)
        num_rays_per_sensor = 8
        angle_step = 2 * math.pi / num_rays_per_sensor
        
        for i in range(16):
            dist = dists[i]
            
            # Determinar se é Trator (0-7) ou Trailer (8-15)
            if i < 8:
                # Sensor 1: Trator
                origin = (ctx, cty)
                base_angle = theta
                local_angle = i * angle_step
                color = (50, 255, 50) # Verde Brilhante
            else:
                # Sensor 2: Trailer
                origin = (ctrx, ctry)
                base_angle = theta_trailer
                local_angle = (i - 8) * angle_step
                color = (50, 255, 255) # Ciano Brilhante
            
            # Ângulo Global do Raio
            global_angle = base_angle + local_angle
            
            # Calcular Ponto Final (Impacto ou Max Range)
            end_x = origin[0] + dist * math.cos(global_angle)
            end_y = origin[1] + dist * math.sin(global_angle)
            
            # Conversão para Tela
            start_px = self.world_to_screen(np.array(origin), focus_pos)
            end_px = self.world_to_screen(np.array([end_x, end_y]), focus_pos)
            
            # Desenhar o Raio
            # Linha fina semi-transparente seria ideal, mas Pygame draw.line é sólida
            pygame.draw.line(self.screen, color, start_px, end_px, 1)
            
            # Desenhar ponto de impacto se colidiu antes do range máximo (ex: < 100m)
            if dist < 99.0: 
                pygame.draw.circle(self.screen, (255, 50, 50), end_px, 3) # Ponto Vermelho

    def draw_info(self, state, controls, z_coords, cost):
        """HUD com informações."""
        texts = [
            f"FPS: {int(pygame.time.Clock().get_fps())}",
            f"State  [x,y]: {state[0]:.2f}, {state[1]:.2f}",
            f"Angles [θ,β]: {math.degrees(state[2]):.1f}°, {math.degrees(state[3]):.1f}°",
            f"Input  [v,α]: {controls[0]:.2f} m/s, {math.degrees(controls[1]):.1f}°",
            "--- Privileged Coords ---",
            f"z1 (Long): {z_coords[0]:.4f}",
            f"z2 (Ornt): {z_coords[1]:.4f}",
            f"z3 (Lat) : {z_coords[2]:.4f}",
            f"z4 (Art) : {z_coords[3]:.4f}",
            f"Tailored Cost: {cost:.4f}",
            "--- Controls ---",
            "W/S: Acelerar | A/D: Esterçar"
        ]
        
        for i, line in enumerate(texts):
            s = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(s, (10, 10 + i * 16))

    def _get_rect_corners(self, x, y, theta, w, l):
        # Cria vértices de um retângulo orientado
        c, s = math.cos(theta), math.sin(theta)
        # length é eixo X local, width é Y local
        dx = l / 2
        dy = w / 2
        corners_local = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        corners_global = []
        for lx, ly in corners_local:
            gx = x + (lx * c - ly * s)
            gy = y + (lx * s + ly * c)
            corners_global.append(np.array([gx, gy]))
        return corners_global

# ==============================================================================
# 3. Simulador (Lógica Principal Ajustada)
# ==============================================================================
class ParkingSimulator:
    def __init__(self):
        self.device = torch.device("cpu") # CPU para visualização simples
        self.dt = 0.05 # Passo de tempo fixo
        
        # 1. Mapa
        self.map_gen = ParkingMapGenerator({'map_width': 200, 'map_height': 200})
        self.map = self.map_gen.generate_map()
        self.obstacles = self.map.get_obstacles_tensor().to(self.device)
        
        # 2. Física e Veículo (AGORA UNIFICADO)
        # Substituímos KinematicModelGPU + RungeKutta4GPU por VehicleDynamicsGPU
        self.dynamics = VehicleDynamicsGPU(
            config=VEHICLE_CONFIG,
            dt=self.dt,
            device=self.device
        )
        
        # Sistemas Auxiliares (Geometria e Sensores permanecem)
        self.geometry = VehicleGeometryGPU(VEHICLE_CONFIG)
        self.collision_sys = CollisionAndSensorSystemGPU(VEHICLE_CONFIG, self.device)
        
        # 3. Estado Inicial
        start = self.map.start_pose
        if start:
            self.state = torch.tensor([[start.position_x, start.position_y, start.theta, 0.0]], device=self.device)
        else:
            self.state = torch.tensor([[50.0, 50.0, 0.0, 0.0]], device=self.device)
            
        self.goal_state = torch.tensor([[
            self.map.parking_goal.position_x,
            self.map.parking_goal.position_y,
            self.map.parking_goal.theta,
            0.0
        ]], device=self.device)

        # Controles
        self.current_v = 0.0
        self.current_alpha = 0.0
        self.max_v = 3.0
        self.max_alpha = math.radians(40)

    def update(self, keys):
        # Entrada do Usuário (Teclado)
        target_v = 0.0
        target_alpha = 0.0
        
        if keys[pygame.K_w]: target_v = self.max_v
        if keys[pygame.K_s]: target_v = -self.max_v
        if keys[pygame.K_a]: target_alpha = self.max_alpha  # Esquerda aumenta angulo
        if keys[pygame.K_d]: target_alpha = -self.max_alpha # Direita diminui
        
        # Suavização simples (Inércia dos atuadores)
        self.current_v += (target_v - self.current_v) * 0.1
        self.current_alpha += (target_alpha - self.current_alpha) * 0.1
        
        control = torch.tensor([[self.current_v, self.current_alpha]], device=self.device)
        
        # --- Passo de Física (Atualizado) ---
        # Chama o step unificado que já faz RK4 internamente
        self.state = self.dynamics.step(self.state, control)
        
        # --- Detecção ---
        verts = self.geometry.compute_vertices(self.state)
        circles = self.collision_sys._compute_collision_circles(self.state) # Método interno auxiliar
        
        # Check colisão (Parede OU Jackknife)
        is_colliding = self.collision_sys.check_collision(self.state, self.obstacles)
        
        # Check específico de Jackknife para debug (opcional)
        is_jackknife = self.collision_sys.check_jackknife(self.state)
        if is_jackknife.item():
            print("ALERTA: JACKKNIFE DETECTADO!")

        # Raycasting LiDAR
        lidar_readings = self.collision_sys.compute_lidar(self.state, self.obstacles)
        
        # Coordenadas Privilegiadas (Usa os params legados do config para manter compatibilidade)
        # Nota: Idealmente, refatorar get_privileged_coordinates para aceitar D e L diretos também
        z = get_privileged_coordinates(
            self.state, self.goal_state, 
            VEHICLE_CONFIG['inv_D'], VEHICLE_CONFIG['inv_L'], VEHICLE_CONFIG['d_axle_kingpin'] # Usando M como k aproximado ou offset_factor
        )
        cost = tailored_cost_function(z, control)
        
        return verts, circles, is_colliding.item(), z[0], cost.item(), lidar_readings[0]
    
    def run(self):
        renderer = ParkingRenderer(100, 100)
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEWHEEL:
                    renderer.ppm += event.y * 0.5
                    renderer.ppm = max(2.0, min(50.0, renderer.ppm))

            keys = pygame.key.get_pressed()
            verts, circles, collision, z, cost, lidar_data = self.update(keys)
            
            focus = self.state[0, :2].cpu().numpy()
            
            renderer.draw_map(self.obstacles, focus)
            
            goal_np = self.goal_state[0].cpu().numpy()
            g_corners = renderer._get_rect_corners(goal_np[0], goal_np[1], goal_np[2], 6.0, 16.0)
            g_poly = [renderer.world_to_screen(p, focus) for p in g_corners]
            pygame.draw.polygon(renderer.screen, (0, 255, 0), g_poly, 3)

            renderer.draw_vehicle(verts, focus, collision)
            renderer.draw_debug_circles(circles, {'tractor_radius': self.collision_sys.tractor_radius, 'trailer_radius': self.collision_sys.trailer_radius}, focus)
            # Desenhar LiDAR Depois do veículo para os raios ficarem "embaixo"
            renderer.draw_lidar(lidar_data, self.state[0], focus)
            state_np = self.state[0].cpu().numpy()
            renderer.draw_info(state_np, [self.current_v, self.current_alpha], z.cpu().numpy(), cost)
            
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    # Certifique-se de que os arquivos auxiliares estão no path ou combine o código
    print("Iniciando Simulador...")
    print(f"Dimensões do Veículo: W={VEHICLE_CONFIG['width_tractor']}m")
    sim = ParkingSimulator()
    sim.run()