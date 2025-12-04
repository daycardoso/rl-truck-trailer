import torch
import math

class CollisionAndSensorSystemGPU:
    """
    Sistema avançado de percepção e colisão para Trator-Trailer.
    
    Funcionalidades:
    1. Detecção de Colisão Física (Discos vs Retângulos).
    2. Detecção de Jackknife (Ângulo de articulação excessivo).
    3. LiDAR Raycasting Analítico (Raio vs Retângulo Rotacionado).
       - Sensores posicionados nos centros geométricos dos componentes.
       - 8 raios por componente.
       - Alcance de 100m.
    """

    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.cfg = config
        
        # --- Configuração LiDAR ---
        self.num_rays_per_sensor = 8
        self.lidar_range = 100.0
        self.jackknife_limit = math.radians(75) # Limite de 90 graus para articulação
        
        # --- Configuração Geométrica (Discos de Colisão) ---
        # Mantemos a lógica dos discos para colisão física pois é muito rápida
        lt = config.get('length_tractor', 5.5)
        wt = config.get('width_tractor', 2.63)
        self.tractor_radius = wt / 2.0
        
        # Offsets dos discos de colisão do Trator (Ref: Eixo Traseiro)
        self.tractor_collision_offsets = torch.tensor([
            [0.0, 0.0],              
            [lt / 2.0, 0.0],         
            [lt - self.tractor_radius, 0.0] 
        ], device=device)

        ltr = config.get('length_trailer', 13.6)
        wtr = config.get('width_trailer', 2.63)
        d_kingpin = config.get('d_front_trailer_kingpin', 1.6)
        
        self.trailer_radius = wtr / 2.0
        dist_pino_fundo = ltr - d_kingpin
        
        # Offsets dos discos de colisão do Trailer (Ref: Pino Rei)
        self.trailer_collision_offsets = torch.tensor([
            [0.5 * d_kingpin, 0.0],     
            [-dist_pino_fundo / 2.0, 0.0], 
            [-dist_pino_fundo + self.trailer_radius, 0.0] 
        ], device=device)

        self.d_axle_kingpin = config.get('d_axle_kingpin', -0.38)
        
        # --- Offsets dos Sensores LiDAR (Centros Geométricos) ---
        # Sensor 1: Centro do Trator (aprox meio do comprimento a partir do eixo)
        # Sensor 2: Centro do Trailer (aprox meio da carga a partir do pino)
        self.sensor_offsets = {
            'tractor': torch.tensor([[lt / 2.0, 0.0]], device=device), # (1, 2)
            'trailer': torch.tensor([[-dist_pino_fundo / 2.0, 0.0]], device=device) # (1, 2)
        }

    # ==========================================================================
    # 1. Detecção de Jackknife (Efeito Canivete)
    # ==========================================================================
    def check_jackknife(self, states: torch.Tensor) -> torch.Tensor:
        """
        Verifica se o ângulo de articulação (beta) excede o limite seguro.
        
        Retorno:
        torch.Tensor (Batch,) -> True se beta > 90 graus
        """
        beta = states[:, 3]
        # Normaliza beta para [-pi, pi] só para garantir, embora o integrador já faça
        beta_norm = torch.atan2(torch.sin(beta), torch.cos(beta))
        return torch.abs(beta_norm) > self.jackknife_limit

    # ==========================================================================
    # 2. Detecção de Colisão Física (Discos)
    # ==========================================================================
    def check_collision(self, states: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        """
        Verifica colisão física + Jackknife.
        Retorna True se bateu na parede OU se deu L.
        """
        # 1. Verifica Jackknife
        is_jackknife = self.check_jackknife(states)
        
        # 2. Verifica Colisão com Obstáculos
        circles = self._compute_collision_circles(states)
        
        all_centers = torch.cat([circles['tractor_centers'], circles['trailer_centers']], dim=1)
        
        radii = torch.cat([
            torch.full((states.shape[0], 3), self.tractor_radius, device=self.device),
            torch.full((states.shape[0], 3), self.trailer_radius, device=self.device)
        ], dim=1)

        if obstacles.shape[0] == 0:
            return is_jackknife # Se não tem parede, só retorna se quebrou o L

        # Lógica Discos vs Retângulos (SDF aproximado)
        c_pos = all_centers.unsqueeze(2) # (Batch, 6, 1, 2)
        obs = obstacles.unsqueeze(0).unsqueeze(0) # (1, 1, Num_Obst, 5)
        
        o_x, o_y = obs[..., 0], obs[..., 1]
        o_theta = obs[..., 2]
        o_w, o_l = obs[..., 3], obs[..., 4]
        
        dx = c_pos[..., 0] - o_x
        dy = c_pos[..., 1] - o_y
        
        cos_o = torch.cos(-o_theta)
        sin_o = torch.sin(-o_theta)
        
        local_x = dx * cos_o - dy * sin_o
        local_y = dx * sin_o + dy * cos_o
        
        extent_x = o_l / 2.0
        extent_y = o_w / 2.0
        
        closest_x = torch.clamp(local_x, -extent_x, extent_x)
        closest_y = torch.clamp(local_y, -extent_y, extent_y)
        
        dist_sq = (local_x - closest_x)**2 + (local_y - closest_y)**2
        
        radii_expanded = radii.unsqueeze(2)
        collision_matrix = dist_sq < (radii_expanded ** 2)
        
        is_physical_collision = collision_matrix.any(dim=2).any(dim=1)
        
        # Retorna True se qualquer condição de falha for atendida
        return is_physical_collision | is_jackknife

    # ==========================================================================
    # 3. LiDAR Raycasting (Slab Method Vetorizado)
    # ==========================================================================
    def compute_lidar(self, states: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        """
        Executa raycasting analítico de 100m.
        
        Config:
        - 8 raios no centro do Trator.
        - 8 raios no centro do Trailer.
        
        Retorno:
        Tensor (Batch, 16) -> Distâncias normalizadas ou em metros.
        """
        batch_size = states.shape[0]
        
        # --- A. Calcular Origens dos Sensores (Centros Geométricos) ---
        x = states[:, 0:1]
        y = states[:, 1:2]
        theta_trator = states[:, 2:3]
        beta = states[:, 3:4]
        theta_trailer = theta_trator + beta
        
        cos_t = torch.cos(theta_trator)
        sin_t = torch.sin(theta_trator)
        
        # Centro Trator Global
        # P_trator = P_eixo + R * offset_centro
        off_t = self.sensor_offsets['tractor'] # (1, 2)
        ctx = x + (cos_t * off_t[:, 0] - sin_t * off_t[:, 1])
        cty = y + (sin_t * off_t[:, 0] + cos_t * off_t[:, 1])
        center_tractor = torch.cat([ctx, cty], dim=1) # (Batch, 2)

        # Centro Trailer Global
        # Quinta Roda
        qx = x + (cos_t * self.d_axle_kingpin)
        qy = y + (sin_t * self.d_axle_kingpin)
        
        cos_tt = torch.cos(theta_trailer)
        sin_tt = torch.sin(theta_trailer)
        
        off_tr = self.sensor_offsets['trailer']
        ctrx = qx + (cos_tt * off_tr[:, 0] - sin_tt * off_tr[:, 1])
        ctry = qy + (sin_tt * off_tr[:, 0] + cos_tt * off_tr[:, 1])
        center_trailer = torch.cat([ctrx, ctry], dim=1) # (Batch, 2)
        
        # Agrupar Origens: (Batch, 2_sensors, 2_coords)
        origins = torch.stack([center_tractor, center_trailer], dim=1)
        
        # --- B. Calcular Direções dos Raios (Globais) ---
        # 8 ângulos fixos (0, 45, 90...) relativos ao veículo
        angles_local = torch.linspace(0, 2*math.pi - (2*math.pi/8), 8, device=self.device)
        
        # Orientação de cada sensor: (Batch, 2_sensors, 1)
        # Sensor 0 (Trator) usa theta_trator, Sensor 1 (Trailer) usa theta_trailer
        sensor_yaws = torch.stack([theta_trator, theta_trailer], dim=1) 
        
        # Broadcasting para obter ângulo global de cada raio
        # (Batch, 2, 1) + (8,) -> (Batch, 2, 8)
        angles_global = sensor_yaws + angles_local.view(1, 1, 8)
        
        # Vetores de Direção (Unitários): (Batch, 2, 8, 2)
        ray_dirs = torch.stack([torch.cos(angles_global), torch.sin(angles_global)], dim=-1)
        
        # Flatten para lista de raios: (Batch, 16, 2)
        # 16 raios no total (8 trator + 8 trailer)
        all_origins = origins.repeat_interleave(8, dim=1) # (Batch, 16, 2)
        all_dirs = ray_dirs.view(batch_size, 16, 2)       # (Batch, 16, 2)
        
        # --- C. Ray vs Box Intersection (Slab Method) ---
        if obstacles.shape[0] == 0:
             return torch.full((batch_size, 16), self.lidar_range, device=self.device)
        
        # Transformar raios para o frame local de CADA obstáculo
        # Obs: (N_obs, 5)
        # Rays: (Batch, 16, 2)
        # Output desejado: Distância mínima para cada raio.
        
        # Expandir: (Batch, 16, N_obs, 2)
        ro_expanded = all_origins.unsqueeze(2) 
        rd_expanded = all_dirs.unsqueeze(2)
        obs_expanded = obstacles.unsqueeze(0).unsqueeze(0) # (1, 1, N_obs, 5)
        
        # Dados do obstáculo
        ox, oy = obs_expanded[..., 0], obs_expanded[..., 1]
        otheta = obs_expanded[..., 2]
        ow, ol = obs_expanded[..., 3], obs_expanded[..., 4] # w=Y, l=X (conforme MapEntity)
        
        # Rotação inversa do obstáculo
        cos_o = torch.cos(-otheta)
        sin_o = torch.sin(-otheta)
        
        # Transladar origem do raio
        diff_x = ro_expanded[..., 0] - ox
        diff_y = ro_expanded[..., 1] - oy
        
        # Rotacionar origem do raio (RO local)
        ro_loc_x = diff_x * cos_o - diff_y * sin_o
        ro_loc_y = diff_x * sin_o + diff_y * cos_o
        
        # Rotacionar direção do raio (RD local)
        rd_loc_x = rd_expanded[..., 0] * cos_o - rd_expanded[..., 1] * sin_o
        rd_loc_y = rd_expanded[..., 0] * sin_o + rd_expanded[..., 1] * cos_o
        
        # Evitar divisão por zero (epsilon)
        eps = 1e-6
        rd_loc_x = torch.where(torch.abs(rd_loc_x) < eps, torch.tensor(eps, device=self.device), rd_loc_x)
        rd_loc_y = torch.where(torch.abs(rd_loc_y) < eps, torch.tensor(eps, device=self.device), rd_loc_y)
        
        # Slab Method (Intersecção com AABB [-l/2, l/2] x [-w/2, w/2])
        # Tempos de intersecção para planos X
        tx1 = (-ol/2.0 - ro_loc_x) / rd_loc_x
        tx2 = (ol/2.0 - ro_loc_x) / rd_loc_x
        tmin_x = torch.min(tx1, tx2)
        tmax_x = torch.max(tx1, tx2)
        
        # Tempos de intersecção para planos Y
        ty1 = (-ow/2.0 - ro_loc_y) / rd_loc_y
        ty2 = (ow/2.0 - ro_loc_y) / rd_loc_y
        tmin_y = torch.min(ty1, ty2)
        tmax_y = torch.max(ty1, ty2)
        
        # Intersecção final do intervalo [t_enter, t_exit]
        t_enter = torch.max(tmin_x, tmin_y)
        t_exit = torch.min(tmax_x, tmax_y)
        
        # Validação:
        # 1. t_exit >= t_enter (raio cruza a caixa)
        # 2. t_exit >= 0 (caixa não está totalmente atrás do raio)
        valid_hit = (t_exit >= t_enter) & (t_exit >= 0)
        
        # Distância é t_enter. Se t_enter < 0, significa que origem está DENTRO da caixa -> dist=0
        dist = torch.where(t_enter < 0, torch.zeros_like(t_enter), t_enter)
        
        # Aplicar Range Máximo onde não houve hit
        dist = torch.where(valid_hit, dist, torch.tensor(float('inf'), device=self.device))
        
        # Pegar a menor distância entre todos os obstáculos para cada raio
        # (Batch, 16, N_obs) -> (Batch, 16)
        min_dist_per_ray, _ = torch.min(dist, dim=2)
        
        # Clamp final para lidar_range (caso seja inf ou > 100)
        lidar_output = torch.clamp(min_dist_per_ray, 0.0, self.lidar_range)
        
        return lidar_output

    def _compute_collision_circles(self, states):
        """Função auxiliar interna (mesma lógica anterior) para calcular centros dos discos."""
        x = states[:, 0:1]
        y = states[:, 1:2]
        theta = states[:, 2:3]
        beta = states[:, 3:4]
        theta_trailer = theta + beta

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        off_x = self.tractor_collision_offsets[:, 0]
        off_y = self.tractor_collision_offsets[:, 1]
        trac_cx = x + (cos_t * off_x - sin_t * off_y)
        trac_cy = y + (sin_t * off_x + cos_t * off_y)
        tractor_centers = torch.stack([trac_cx, trac_cy], dim=2)

        qx = x + (cos_t * self.d_axle_kingpin)
        qy = y + (sin_t * self.d_axle_kingpin)

        cos_tt = torch.cos(theta_trailer)
        sin_tt = torch.sin(theta_trailer)
        
        off_trx = self.trailer_collision_offsets[:, 0]
        off_try = self.trailer_collision_offsets[:, 1]
        trail_cx = qx + (cos_tt * off_trx - sin_tt * off_try)
        trail_cy = qy + (sin_tt * off_trx + cos_tt * off_try)
        trailer_centers = torch.stack([trail_cx, trail_cy], dim=2)

        return {'tractor_centers': tractor_centers, 'trailer_centers': trailer_centers}