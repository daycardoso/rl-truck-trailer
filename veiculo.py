import torch
import math

class VehicleDynamicsGPU:
    """
    Simulador físico unificado para Trator-Trailer (Modelo + Integrador RK4).
    
    Implementação vetorizada em PyTorch baseada no modelo de referência 'TratorComUmTrailer'.
    Combina as equações diferenciais cinemáticas com o método de Runge-Kutta de 4ª ordem.

    Estado (state): [x, y, theta, beta]
        - x, y: Coordenadas do ponto médio do eixo traseiro do trator.
        - theta: Ângulo de orientação do trator.
        - beta: Ângulo relativo de articulação (theta_trailer - theta_trator).

    Controle (action): [v, alpha]
        - v: Velocidade linear longitudinal do trator.
        - alpha: Ângulo de esterçamento das rodas dianteiras.
    """

    def __init__(self, config: dict, dt: float = 0.1, device: str = "cpu"):
        self.dt = dt
        self.device = torch.device(device)

        # 1. Extração de Parâmetros Físicos (Baseado no modelo confiável)
        # D: Distância entre eixos do trator
        self.D = config.get('length_tractor_wheelbase', 3.29) 
        
        # L: Distância eixo traseiro trailer -> quinta roda
        self.L = config.get('length_trailer_wheelbase', 7.135) 
        
        # M: Distância eixo traseiro trator -> quinta roda (quinta_roda)
        self.M = config.get('d_axle_kingpin', -0.38)

        # 2. Pré-cálculo de constantes (Para performance)
        # Evita divisões no loop de simulação
        self.inv_D = 1.0 / self.D if self.D != 0.0 else 0.0
        self.inv_L = 1.0 / self.L if self.L != 0.0 else 0.0
        
        # Fator de deslocamento exato do modelo de referência:
        # self.offset_factor = self.quinta_roda * self.inv_L * self.inv_D
        self.offset_factor = self.M * self.inv_L * self.inv_D

    def step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Realiza um passo de simulação completo (Integração RK4).

        Args:
            state: Tensor (Batch, 4) -> [x, y, theta, beta]
            control: Tensor (Batch, 2) -> [v, alpha]

        Returns:
            next_state: Tensor (Batch, 4) atualizado e normalizado.
        """
        # RK4 Integration Steps
        # k1 = f(x, u)
        k1 = self._compute_derivatives(state, control)

        # k2 = f(x + 0.5*dt*k1, u)
        k2 = self._compute_derivatives(state + (0.5 * self.dt * k1), control)

        # k3 = f(x + 0.5*dt*k2, u)
        k3 = self._compute_derivatives(state + (0.5 * self.dt * k2), control)

        # k4 = f(x + dt*k3, u)
        k4 = self._compute_derivatives(state + (self.dt * k3), control)

        # x_{n+1} = x_n + (dt/6) * (k1 + 2k2 + 2k3 + k4)
        delta = (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state = state + delta

        # Normalização dos ângulos (theta e beta) para [-pi, pi]
        next_state = self._normalize_angles(next_state)

        return next_state

    def _compute_derivatives(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Calcula q_dot = f(q, u) seguindo as equações exatas do modelo confiável.
        """
        # Desempacota estado
        # x = state[:, 0] # Não usado na derivada, apenas na integração
        # y = state[:, 1]
        theta = state[:, 2]
        beta = state[:, 3]

        # Desempacota controle
        v = control[:, 0]
        alpha = control[:, 1]

        # Cálculos Trigonométricos
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        tan_alpha = torch.tan(alpha)
        sin_beta = torch.sin(beta)
        cos_beta = torch.cos(beta)

        # --- Equações de Movimento (Idênticas ao TratorComUmTrailer) ---
        
        # ẋ = v * cos(theta)
        dx = v * cos_theta
        
        # ẏ = v * sin(theta)
        dy = v * sin_theta
        
        # thetȧ = (v/D) * tan(alpha)
        dtheta = v * self.inv_D * tan_alpha
        
        # betȧ = - (v/L * sin(beta)) - (v/D * tan(alpha)) + (offset_factor * v * tan(alpha) * cos(beta))
        # Termo 1: Efeito do arrasto do trailer
        term1 = - (v * self.inv_L * sin_beta)
        # Termo 2: Efeito da rotação do trator sobre o engate
        term2 = - (v * self.inv_D * tan_alpha)
        # Termo 3: Correção geométrica do deslocamento da quinta roda
        term3 = (self.offset_factor * v * tan_alpha * cos_beta)
        
        dbeta = term1 + term2 + term3

        # Empilha as derivadas para retornar (Batch, 4)
        return torch.stack([dx, dy, dtheta, dbeta], dim=1)

    def _normalize_angles(self, state: torch.Tensor) -> torch.Tensor:
        """Mantém theta e beta dentro de [-pi, pi] usando atan2."""
        # Clona para evitar inplace operation perigosa no grafo do PyTorch
        new_state = state.clone()
        
        # Theta (idx 2)
        new_state[:, 2] = torch.atan2(torch.sin(new_state[:, 2]), torch.cos(new_state[:, 2]))
        
        # Beta (idx 3)
        new_state[:, 3] = torch.atan2(torch.sin(new_state[:, 3]), torch.cos(new_state[:, 3]))
        
        return new_state

# ==============================================================================
# Exemplo de Configuração e Uso (Teste Unitário)
# ==============================================================================
if __name__ == "__main__":
    # Parâmetros baseados nos seus dados reais
    config_real = {
        'length_tractor_wheelbase': 3.29,  # D
        'length_trailer_wheelbase': 7.135, # L
        'd_axle_kingpin': -0.38            # M (quinta_roda)
    }

    # Inicializa
    dynamics = VehicleDynamicsGPU(config_real, dt=0.05)

    # Estado Inicial: [x=0, y=0, theta=0, beta=0]
    # Batch de 2 veículos para teste
    state0 = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],          # Veículo 1: Reto
        [0.0, 0.0, 0.0, 0.1]           # Veículo 2: Levemente articulado
    ])

    # Controle: [v=5 m/s, alpha=10 graus]
    alpha_rad = math.radians(10)
    control = torch.tensor([
        [5.0, alpha_rad],
        [5.0, alpha_rad]
    ])

    # Passo de Simulação
    next_state = dynamics.step(state0, control)

    print(f"--- Teste de Física (dt={dynamics.dt}s) ---")
    print(f"Inputs D={dynamics.D}, L={dynamics.L}, M={dynamics.M}")
    print(f"Offset Factor calculado: {dynamics.offset_factor:.6f}")
    print("\nEstado Inicial:")
    print(state0)
    print("\nEstado Seguinte (RK4):")
    print(next_state)
    
    # Validação rápida de movimento
    # Se x aumentou e y aumentou (devido a alpha positivo), está fazendo curva à esquerda
    if next_state[0, 0] > 0 and next_state[0, 1] > 0:
        print("\n✅ Validação: Veículo moveu-se para frente e para a esquerda conforme esperado.")