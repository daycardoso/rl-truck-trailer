import numpy as np
import torch
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class MapEntity:
    """Representa um objeto estático no mapa (Parede, Vaga, Obstáculo)."""
    

    position_x: float
    position_y: float
    width: float  # Dimensão no eixo Y local
    length: float # Dimensão no eixo X local
    theta: float
    type: int
    
    # Tipos de Entidade
    ENTITY_WALL: int = 0
    ENTITY_PARKING_SLOT: int = 1
    ENTITY_PARKING_GOAL: int = 2
    ENTITY_START: int = 3
    ENTITY_OBSTACLE: int = 4 # Carros estacionados (agem como paredes)
    
    def get_tensor_data(self):
        """Retorna representação [x, y, theta, width, length] para GPU."""
        return torch.tensor([
            self.position_x, self.position_y, self.theta, self.width, self.length
        ], dtype=torch.float32)

class Map:
    """Container para todos os elementos do ambiente."""
    
    def __init__(self, dimensions: Tuple[float, float]):
        self.width, self.height = dimensions
        self.entities: List[MapEntity] = []
        self.parking_slots: List[MapEntity] = []
        
        self.parking_goal: Optional[MapEntity] = None
        self.start_pose: Optional[MapEntity] = None # Define onde o trator nasce
        
    def add_entity(self, entity: MapEntity):
        self.entities.append(entity)
        if entity.type in [MapEntity.ENTITY_PARKING_SLOT, MapEntity.ENTITY_PARKING_GOAL]:
            self.parking_slots.append(entity)
            
    def get_obstacles_tensor(self) -> torch.Tensor:
        """
        Retorna um tensor contendo todas as paredes e obstáculos.
        Usado para cálculo vetorizado de colisão.
        Shape: (N_obstacles, 5) -> [x, y, theta, width, length]
        """
        obstacles = [
            e.get_tensor_data() for e in self.entities 
            if e.type in [MapEntity.ENTITY_WALL, MapEntity.ENTITY_OBSTACLE]
        ]
        if not obstacles:
            return torch.empty((0, 5))
        return torch.stack(obstacles)

    def get_random_parking_slot(self) -> MapEntity:
        return random.choice(self.parking_slots)

    def get_parking_slots(self) -> List[MapEntity]:
        return self.parking_slots

class ParkingMapGenerator:
    """
    Gera um layout de estacionamento com paredes, vagas e obstáculos aleatórios.
    """
    
    def __init__(self, config: dict = None):
        self.cfg = config or {}
        
        # Parâmetros Geométricos (Default ou Config)
        self.MAP_WIDTH = self.cfg.get('map_width', 250.0)
        self.MAP_HEIGHT = self.cfg.get('map_height', 250.0)
        self.WALL_WIDTH = 4.0
        
        # Vagas (Ajustado para caber um caminhão/carreta?)
        # Vagas de carro costumam ser 2.5x5.0m. Para caminhão precisamos de mais.
        self.PARKING_SLOT_WIDTH = 6.0 
        self.PARKING_SLOT_HEIGHT = 14.0 
        
        self.SPAWN_PADDING = 25.0
        self.WALL_PADDING = 3.0
        self.N_ROWS = 2 # Fileira superior e inferior

    def generate_map(self) -> Map:
        game_map = Map((self.MAP_WIDTH, self.MAP_HEIGHT))
        
        # --- 1. Paredes de Contorno ---
        # Nota: Ajustamos position para width/2 para a parede ficar "dentro" do limite lógico
        
        # Esquerda
        game_map.add_entity(MapEntity(
            position_x=self.WALL_WIDTH/2, position_y=self.MAP_HEIGHT/2, 
            width=self.WALL_WIDTH, length=self.MAP_HEIGHT, 
            theta=math.pi/2, type=MapEntity.ENTITY_WALL
        ))
        # Direita
        game_map.add_entity(MapEntity(
            position_x=self.MAP_WIDTH - self.WALL_WIDTH/2, position_y=self.MAP_HEIGHT/2, 
            width=self.WALL_WIDTH, length=self.MAP_HEIGHT, 
            theta=math.pi/2, type=MapEntity.ENTITY_WALL # Mantive pi/2 para alinhar verticalmente
        ))
        # Topo
        game_map.add_entity(MapEntity(
            position_x=self.MAP_WIDTH/2, position_y=self.WALL_WIDTH/2, 
            width=self.WALL_WIDTH, length=self.MAP_WIDTH, 
            theta=0.0, type=MapEntity.ENTITY_WALL
        ))
        # Base
        game_map.add_entity(MapEntity(
            position_x=self.MAP_WIDTH/2, position_y=self.MAP_HEIGHT - self.WALL_WIDTH/2, 
            width=self.WALL_WIDTH, length=self.MAP_WIDTH, 
            theta=0.0, type=MapEntity.ENTITY_WALL
        ))

        # --- 2. Layout das Fileiras ---
        # Define alturas para as ilhas de vagas
        top_row_y = self.MAP_HEIGHT / 3.0
        bottom_row_y = 2.0 * self.MAP_HEIGHT / 3.0
        
        area_width = self.MAP_WIDTH - 2 * self.SPAWN_PADDING
        
        # Parede Central Superior (Backstop das vagas de cima)
        game_map.add_entity(MapEntity(
            position_x=self.MAP_WIDTH/2, position_y=top_row_y, 
            width=1.0, length=area_width, # Parede fina
            theta=0.0, type=MapEntity.ENTITY_WALL
        ))
        
        # Parede Central Inferior (Backstop das vagas de baixo)
        game_map.add_entity(MapEntity(
            position_x=self.MAP_WIDTH/2, position_y=bottom_row_y, 
            width=1.0, length=area_width, 
            theta=0.0, type=MapEntity.ENTITY_WALL
        ))

        # --- 3. Geração das Vagas ---
        start_x = self.SPAWN_PADDING + self.PARKING_SLOT_WIDTH/2
        end_x = self.MAP_WIDTH - self.SPAWN_PADDING - self.PARKING_SLOT_WIDTH/2
        
        current_x = start_x
        while current_x <= end_x:
            # Vaga Superior (Apontando para baixo ou perpendicular)
            # Centro da vaga deve estar deslocado do muro central
            offset = (self.PARKING_SLOT_HEIGHT / 2.0) + self.WALL_PADDING
            
            slot_up = MapEntity(
                position_x=current_x,
                position_y=top_row_y - offset,
                width=self.PARKING_SLOT_WIDTH,
                length=self.PARKING_SLOT_HEIGHT,
                theta=-math.pi/2, # Apontando para baixo (entrada pelo sul da vaga)
                type=MapEntity.ENTITY_PARKING_SLOT
            )
            game_map.add_entity(slot_up)
            
            # Vaga Inferior
            slot_down = MapEntity(
                position_x=current_x,
                position_y=bottom_row_y + offset,
                width=self.PARKING_SLOT_WIDTH,
                length=self.PARKING_SLOT_HEIGHT,
                theta=math.pi/2, # Apontando para cima (entrada pelo norte da vaga)
                type=MapEntity.ENTITY_PARKING_SLOT
            )
            game_map.add_entity(slot_down)
            
            current_x += self.PARKING_SLOT_WIDTH

        # --- 4. Lógica de Missão (Start / Goal) ---
        slots = game_map.get_parking_slots()
        
        # Escolhe Objetivo
        goal_slot = random.choice(slots)
        goal_slot.type = MapEntity.ENTITY_PARKING_GOAL
        game_map.parking_goal = goal_slot
        
        # Escolhe Partida (Garantindo que não é o mesmo)
        available_starts = [s for s in slots if s != goal_slot]
        start_entity = random.choice(available_starts)
        
        # O "MapEntity" de start define a POSE inicial do veículo.
        # Aqui, colocamos o veículo estacionado em outra vaga (cenário de manobra de saída/troca)
        # Se quiser nascer no corredor (aisle), crie uma entidade no meio do mapa.
        start_entity.type = MapEntity.ENTITY_START 
        game_map.start_pose = start_entity

        # --- 5. Obstáculos (Carros Estacionados) ---
        for slot in slots:
            # Se não for Start nem Goal, tem chance de ter um carro
            if slot.type == MapEntity.ENTITY_PARKING_SLOT:
                if np.random.random() < 0.4: # 40% de ocupação
                    # Adiciona "parede" (obstáculo) dentro da vaga
                    # Dimensões levemente menores que a vaga
                    car_obstacle = MapEntity(
                        position_x=slot.position_x,
                        position_y=slot.position_y,
                        width=slot.width - 1.0,   # Folga lateral
                        length=slot.length - 2.0, # Folga longitudinal
                        theta=slot.theta,
                        type=MapEntity.ENTITY_OBSTACLE
                    )
                    game_map.add_entity(car_obstacle)

        return game_map