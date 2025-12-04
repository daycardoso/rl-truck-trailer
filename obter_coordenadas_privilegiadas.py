import torch

def get_privileged_coordinates(
    current_state: torch.Tensor,
    goal_state: torch.Tensor,
    inv_D: float,
    inv_L: float,
    k: float
) -> torch.Tensor:
    """
    Transforma o erro de estado global nas Coordenadas Privilegiadas (z).

    Esta função implementa a **Etapa 3** do Design Procedure[cite: 10], convertendo
    o erro do sistema para uma base onde a controlabilidade é explícita.
    
    A transformação segue dois passos:
    1. Mudança de referencial: O erro global é rotacionado para o frame do objetivo.
    2. Transformação não-holonômica: Aplicação das equações simbólicas derivadas.

    Equações Derivadas (para setpoint na origem):
    
    .. math::
        z_1 = x
        z_2 = \\frac{\\theta}{\\text{inv\\_D}}
        z_3 = -\\frac{y}{\\text{inv\\_D}}
        z_4 = \\frac{-\\beta \\cdot \\text{inv\\_D} + \\text{inv\\_L} \\cdot y \\cdot (\\text{inv\\_D} - k) - \\theta \\cdot (\\text{inv\\_D} - k)}{\\text{inv\\_D} \\cdot \\text{inv\\_L}^2 \\cdot (\\text{inv\\_D} - k)}

    Parâmetros
    ----------
    current_state : torch.Tensor
        Estado atual ``(Batch, 4)`` contendo ``[x, y, theta, beta]``.
    goal_state : torch.Tensor
        Estado objetivo ``(Batch, 4)`` contendo ``[x_g, y_g, theta_g, beta_g]``.
    inv_D : float
        Inverso da distância entre eixos do trator (1/D).
    inv_L : float
        Inverso do comprimento do reboque (1/L).
    k : float
        Fator de deslocamento do engate (offset factor).

    Retorno
    -------
    torch.Tensor
        Tensor ``(Batch, 4)`` contendo as coordenadas privilegiadas ``[z1, z2, z3, z4]``.
        Estas coordenadas devem ser utilizadas na função de custo tailored.

    Notas
    -----
    A simplificação para setpoint na origem é válida aqui pois rotacionamos o erro
    previamente[cite: 11].
    """
    # 1. Cálculo do Erro no Referencial Global
    dx = current_state[:, 0] - goal_state[:, 0]
    dy = current_state[:, 1] - goal_state[:, 1]
    
    # Normalização de ângulos para evitar descontinuidades em +/- PI
    theta_error = current_state[:, 2] - goal_state[:, 2]
    theta_error = torch.atan2(torch.sin(theta_error), torch.cos(theta_error))
    
    beta_error = current_state[:, 3] - goal_state[:, 3]
    beta_error = torch.atan2(torch.sin(beta_error), torch.cos(beta_error))

    # 2. Rotação para o Referencial do Objetivo (Error Body Frame)
    # Precisamos projetar dx e dy na orientação do objetivo (theta_g)
    theta_g = goal_state[:, 2]
    cos_g = torch.cos(theta_g)
    sin_g = torch.sin(theta_g)

    x_local = dx * cos_g + dy * sin_g
    y_local = -dx * sin_g + dy * cos_g

    # 3. Cálculo das Coordenadas Privilegiadas (z)
    # Usando as variáveis locais (x_local, y_local) como (x, y) das fórmulas
    
    # z1 = x
    z1 = x_local
    
    # z2 = theta / inv_D
    z2 = theta_error / inv_D
    
    # z3 = -y / inv_D
    z3 = -y_local / inv_D
    
    # z4: Implementação da fórmula complexa derivada simbolicamente
    # Termo comum: (inv_D - k)
    delta_Dk = inv_D - k
    
    # Numerador: -beta*inv_D + inv_L*y*(inv_D - k) - theta*(inv_D - k)
    # Nota: theta aqui é o theta_error e y é o y_local
    num_z4 = (-beta_error * inv_D) + \
             (inv_L * y_local * delta_Dk) - \
             (theta_error * delta_Dk)
             
    # Denominador: inv_D * inv_L**2 * (inv_D - k)
    den_z4 = inv_D * (inv_L ** 2) * delta_Dk
    
    # Evitar divisão por zero caso k seja exatamente igual a inv_D
    # (Embora geometricamente improvável em veículos reais)
    epsilon = 1e-6
    z4 = num_z4 / (den_z4 + epsilon)

    # Empilhamento final
    z = torch.stack([z1, z2, z3, z4], dim=1)
    
    return z