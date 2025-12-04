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

    Expoentes calculados via MMC(1, 1, 2, 3) * 2:
        * Longitudinal ($z_1$): 12
        * Orientação ($z_2$): 12
        * Lateral ($z_3$): 6
        * Articulação ($z_4$): 4

    Parâmetros
    ----------
    z : torch.Tensor
        Tensor de formato ``(Batch, 4)`` contendo as coordenadas privilegiadas
        [z1, z2, z3, z4].
    u : torch.Tensor
        Tensor de formato ``(Batch, 2)`` contendo os controles [v, omega].
    weights : torch.Tensor, opcional
        Pesos lineares [lambda1...4] para ajuste fino. Default: ones.

    Retorno
    -------
    torch.Tensor
        Custo escalar para cada elemento do batch ``(Batch,)``.
    
    Exemplo
    -------
    .. code-block:: python

        # Batch de 10 estados
        z = get_privileged_coordinates(current_state, goal_state)
        u = current_controls
        cost = tailored_cost_function(z, u)
    """
    if weights is None:
        weights = torch.ones(4, device=z.device)

    # Cálculo dos termos de estado com expoentes adaptativos
    # z1 e z2 (fáceis) -> potência 12 (muito plano perto de 0, parede vertical longe)
    c_z1 = weights[0] * (z[:, 0] ** 12)
    c_z2 = weights[1] * (z[:, 1] ** 12)
    
    # z3 (médio) -> potência 6
    c_z3 = weights[2] * (z[:, 2] ** 6)
    
    # z4 (difícil - articulação) -> potência 4 (domina o custo quando z < 1)
    c_z4 = weights[3] * (z[:, 3] ** 4)

    # Custo de Controle (Regularização)
    # Potência alta força controles a serem pequenos, mas permite picos breves se necessário
    c_u = torch.sum(u ** 12, dim=1)

    return c_z1 + c_z2 + c_z3 + c_z4 + c_u
