import torch
from typing import Optional
def tailored_cost_function(z: torch.Tensor, u: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calcula o custo 'tailored' (sob medida) para o veículo não-holonômico.

    Utiliza expoentes derivados do grau de homogeneidade $d=12$ para penalizar
    o erro nas coordenadas privilegiadas, respeitando a dificuldade de controle
    de cada estado.

    .. math::
        \\ell(z, u) = \\sum q_i z_i^{d/r_i} + \\sum r_j u_j^{d/s_j}
        
    Calcula a Norma Homogênea (Distance-Like Function) para o veículo.

    Diferente do custo tailored puro (que é polinomial de grau 12), esta função
    retorna a raiz d-ésima da soma ponderada. Isso lineariza o gradiente longe
    da origem, comportando-se como uma métrica de distância, ideal para RL.

    .. math::
        \\rho(z, u) = \\left( \\sum q_i |z_i|^{d/r_i} + \\sum r_j |u_j|^{d/s_j} \\right)^{1/d}

    Expoentes calculados via MMC(1, 1, 2, 3) * 2:
        * Longitudinal ($z_1$): 12
        * Orientação ($z_2$): 12
        * Lateral ($z_3$): 6
        * Articulação ($z_4$): 4


    Parâmetros
    ----------
    z : torch.Tensor
        Coordenadas privilegiadas [z1, z2, z3, z4] (Batch, 4).
    u : torch.Tensor
        Controles [v, alpha] (Batch, 2).
    weights : torch.Tensor, opcional
        Pesos [q1, q2, q3, q4] para os estados. 
        Sugestão: Aumentar q3 e q4 para penalizar mais o desvio lateral/angular.

    Retorno
    -------
    torch.Tensor
        Valor escalar da norma homogênea para cada elemento do batch (Batch,).
        Para usar como recompensa em RL: reward = -rho
    """
    if weights is None:
        # Pesos padrão sugeridos para RL:
        # Prioriza z3 (lateral) e z4 (articulação) que são os "difíceis"
        weights = torch.tensor([1.0, 1.0, 5.0, 10.0], device=z.device)

    # 1. Termos de Estado (State Cost)
    # d=12, r=(1, 1, 2, 3)
    # Expoentes: 12, 12, 6, 4
    
    # Adicionamos epsilon para estabilidade numérica na raiz e gradientes
    epsilon = 1e-6 

    c_z1 = weights[0] * (torch.abs(z[:, 0]) ** 12)
    c_z2 = weights[1] * (torch.abs(z[:, 1]) ** 12)
    c_z3 = weights[2] * (torch.abs(z[:, 2]) ** 6)
    c_z4 = weights[3] * (torch.abs(z[:, 3]) ** 4)

    state_cost = c_z1 + c_z2 + c_z3 + c_z4

    # 2. Termos de Controle (Control Regularization)
    # d=12, s=(1, 1) -> Expoente 12
    # Isso penaliza comandos muito bruscos, suavizando a política
    # Peso pequeno (0.1) para não dominar a necessidade de chegar no alvo
    ctrl_weight = 0.1
    c_u = ctrl_weight * torch.sum(torch.abs(u) ** 12, dim=1)

    # 3. Norma Homogênea
    # rho = (cost)^(1/12)
    # A propriedade de homogeneidade garante que rho(lambda*z) = lambda * rho(z)
    total_cost_poly = state_cost + c_u + epsilon
    homogeneous_norm = torch.pow(total_cost_poly, 1.0 / 12.0)

    return homogeneous_norm 
