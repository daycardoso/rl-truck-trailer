import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
import pygame

# Importando seus componentes
from veiculo import VehicleDynamicsGPU
from geometria_veiculo import VehicleGeometryGPU
from mapa import ParkingMapGenerator, MapEntity
from colisao_sensores import CollisionAndSensorSystemGPU
from obter_coordenadas_privilegiadas import get_privileged_coordinates
from toilered_custo import tailored_cost_function
from main_visualizer import ParkingRenderer, VEHICLE_CONFIG # Reutiliza config e renderer

class TrailerDockingEnv(gym.Env):
    """
    Ambiente Gymnasium para Estacionamento de Trator-Trailer Não-Holonômico.
    
    Integra:
    - Dinâmica RK4 vetorizada.
    - Custo Tailored (Norma Homogênea).
    - Percepção LiDAR analítica.
    - Detecção de Colisão e Jackknife.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, config=None):
        super(TrailerDockingEnv, self).__init__()
        
        self.cfg = config if config else VEHICLE_CONFIG
        self.render_mode = render_mode
        self.device = torch.device("cpu") # RL geralmente roda CPU-bound no step-by-step
        
        # 1. Componentes do Sistema
        self.map_gen = ParkingMapGenerator({'map_width': 100, 'map_height': 100})
        
        self.dynamics = VehicleDynamicsGPU(self.cfg, dt=0.05, device=self.device)
        self.collision_sys = CollisionAndSensorSystemGPU(self.cfg, self.device)
        self.geometry = VehicleGeometryGPU(self.cfg) # Apenas para render
        
        # 2. Espaços de Ação e Observação
        # Ação: [velocidade, esterçamento] normalizados em [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Limites físicos reais para desnormalização
        self.max_v = 5.0  # m/s
        self.max_alpha = math.radians(28) # ~0.7 rad
        
        # Observação (26 dimensões):
        # [State(4) + Privileged(4) + LiDAR(16) + LastAction(2)]
        # Usamos np.inf pois as coordenadas podem crescer, mas na prática são limitadas pelo mapa
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        
        # 3. Estado Interno
        self.state = None      # Tensor (1, 4)
        self.goal = None       # Tensor (1, 4)
        self.obstacles = None  # Tensor (N, 5)
        self.last_action = None # Tensor (1, 2)
        self.steps = 0
        self.max_steps = 600   # Horizonte de tempo
        
        # Renderização
        self.renderer = None
        self.window = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Gerar Novo Mapa e Obstáculos
        game_map = self.map_gen.generate_map()
        self.obstacles = game_map.get_obstacles_tensor().to(self.device)
        
        # 2. Definir Posições (Batch size = 1 para compatibilidade com classes GPU)
        # Start
        start_node = game_map.start_pose
        if start_node:
            # Adiciona ruído leve na inicialização para robustez
            # noise = (torch.rand(4) - 0.5) * 0.2 # +/- 0.1m/rad
            s_tens = torch.tensor([start_node.position_x, start_node.position_y, start_node.theta, 0.0])
            self.state = (s_tens).unsqueeze(0).to(self.device)
        else:
            self.state = torch.tensor([[50.0, 50.0, 0.0, 0.0]]).to(self.device)

        # Goal
        goal_node = game_map.parking_goal
        self.goal = torch.tensor([[
            goal_node.position_x, goal_node.position_y, goal_node.theta, 0.0
        ]]).to(self.device)
        
        # Reset variáveis
        self.last_action = torch.zeros((1, 2)).to(self.device)
        self.steps = 0
        
        # Retorna observação inicial
        return self._get_obs(), {}

    def step(self, action):
        # 1. Desnormalizar Ação (Numpy -> Tensor)
        # action vem do agente como np.array [-1, 1]
        action_climped = np.clip(action, -1.0, 1.0)
        
        # Mapear para físico
        v_cmd = action_climped[0] * self.max_v
        alpha_cmd = action_climped[1] * self.max_alpha
        
        control_tensor = torch.tensor([[v_cmd, alpha_cmd]], dtype=torch.float32).to(self.device)
        
        # 2. Dinâmica (Physics Step)
        self.state = self.dynamics.step(self.state, control_tensor)
        
        # 3. Verificações (Colisão, Jackknife, LiDAR)
        # Check Collision retorna True se bater ou se Jackknife (beta > limite)
        collision_or_jackknife = self.collision_sys.check_collision(self.state, self.obstacles)
        
        lidar_readings = self.collision_sys.compute_lidar(self.state, self.obstacles)
        
        # 4. Cálculo de Coordenadas Privilegiadas (Z)
        z = get_privileged_coordinates(
            self.state, self.goal,
            self.cfg['inv_D'], self.cfg['inv_L'], self.cfg['d_axle_kingpin']
        )
        
        # 5. Cálculo de Recompensa (Norma Homogênea)
        # tailored_cost_function retorna a norma (distância). Queremos maximizar a negativa.
        rho = tailored_cost_function(z, control_tensor) # Escalar positivo
        
        reward = -rho.item()
        
        # Penalidade de Ação (Suavidade) - pequena penalidade quadrática no controle
        reward -= 0.05 * np.sum(action_climped**2)
        
        # Penalidades Terminais e Bônus
        terminated = False
        truncated = False
        
        if collision_or_jackknife.item():
            reward -= 200.0 # Penalidade Alta
            terminated = True
            
        # Sucesso: Se a norma homogênea for muito pequena (< 0.5 é um bom threshold inicial)
        if rho.item() < 0.4:
            reward += 100.0
            terminated = True
            # print("SUCCESS! Parked.")

        # Timeout
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            
        self.last_action = control_tensor
        
        # 6. Observação
        obs = self._get_obs(z, lidar_readings)
        
        return obs, reward, terminated, truncated, {"rho": rho.item()}

    def _get_obs(self, z=None, lidar=None):
        """Monta o vetor de observação numpy."""
        if z is None:
             z = get_privileged_coordinates(
                self.state, self.goal,
                self.cfg['inv_D'], self.cfg['inv_L'], self.cfg['d_axle_kingpin']
            )
        if lidar is None:
            lidar = self.collision_sys.compute_lidar(self.state, self.obstacles)

        # Normalização relativa ao objetivo para o estado global
        # (Opcional, mas ajuda a rede: delta_x, delta_y, etc)
        # Por enquanto, passamos estado bruto, pois Z já codifica o erro relativo.
        
        # Concatenar tudo em um vetor plano
        # State(4) + Z(4) + LiDAR(16) + LastAction(2)
        obs_tensor = torch.cat([
            self.state.squeeze(0),      # 4
            z.squeeze(0),               # 4
            lidar.squeeze(0) / 100.0,   # 16 (Normaliza LiDAR 0-100m -> 0-1)
            self.last_action.squeeze(0) # 2
        ])
        
        return obs_tensor.cpu().numpy().astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = ParkingRenderer(100, 100) # Mapa 100x100
            
            # Recalcular dados necessários para render
            # (Poderíamos cachear no step, mas render é chamado menos vezes)
            verts = self.geometry.compute_vertices(self.state)
            circles = self.collision_sys._compute_collision_circles(self.state)
            lidar = self.collision_sys.compute_lidar(self.state, self.obstacles)
            z = get_privileged_coordinates(
                self.state, self.goal,
                self.cfg['inv_D'], self.cfg['inv_L'], self.cfg['d_axle_kingpin']
            )
            cost = tailored_cost_function(z, self.last_action).item()
            is_col = self.collision_sys.check_collision(self.state, self.obstacles)
            
            # Dados para Pygame
            focus = self.state[0, :2].cpu().numpy()
            
            # Chamar métodos de desenho
            self.renderer.draw_map(self.obstacles, focus)
            
            # Goal
            g_np = self.goal[0].cpu().numpy()
            g_corners = self.renderer._get_rect_corners(g_np[0], g_np[1], g_np[2], 6.0, 16.0)
            g_poly = [self.renderer.world_to_screen(p, focus) for p in g_corners]
            pygame.draw.polygon(self.renderer.screen, (0, 255, 0), g_poly, 3)
            
            # Veículo e Sensores
            self.renderer.draw_lidar(lidar[0], self.state[0], focus)
            self.renderer.draw_vehicle(verts, focus, is_col.item())
            self.renderer.draw_debug_circles(circles, {'tractor_radius': self.collision_sys.tractor_radius, 'trailer_radius': self.collision_sys.trailer_radius}, focus)
            
            # Info
            st_np = self.state[0].cpu().numpy()
            ctl_np = self.last_action[0].cpu().numpy()
            self.renderer.draw_info(st_np, ctl_np, z[0].cpu().numpy(), cost)
            
            pygame.display.flip()
            
            # Processar eventos básicos para não travar janela
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

    def close(self):
        if self.renderer:
            pygame.quit()
            self.renderer = None