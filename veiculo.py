import torch
import math

class KinematicModelGPU:
    """
    Representação em GPU do modelo cinemático de trator-reboque.

    Esta classe implementa a forma matricial Control Affine para processamento
    em lote (batch processing).

    :param inv_D: Inverso da distância entre eixos do trator (1/D).
    :type inv_D: float
    :param inv_L: Inverso do comprimento do reboque (1/L).
    :type inv_L: float
    :param offset_factor: Fator de deslocamento do engate (k).
    :type offset_factor: float
    """

    def __init__(self, inv_D: float, inv_L: float, offset_factor: float):
        self.inv_D = inv_D
        self.inv_L = inv_L
        self.offset_factor = offset_factor

    def compute_derivatives(self, states: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calcula as derivadas temporais do estado (q_dot) para um lote de veículos.

        A computação é realizada via multiplicação de tensores:
        .. math::
            \\dot{\\mathbf{q}} = M(\\mathbf{q}) \\cdot \\mathbf{u}

        Parâmetros
        ----------
        states : torch.Tensor
            Tensor de formato ``(Batch, 4)`` contendo ``[x, y, theta, beta]``.
        inputs : torch.Tensor
            Tensor de formato ``(Batch, 2)`` contendo ``[v, alpha]``. Note que
            este é o input bruto, que será transformado internamente.

        Retorno
        -------
        torch.Tensor
            Derivadas ``(Batch, 4)`` contendo ``[dx, dy, dtheta, dbeta]``.

        Exemplo
        -------
        .. code-block:: python

            model = KinematicModelGPU(inv_D=0.5, inv_L=0.3, offset_factor=0.1)
            # Batch de 1000 veículos
            states = torch.zeros(1000, 4).cuda()
            inputs = torch.tensor([10.0, 0.5]).repeat(1000, 1).cuda()
            
            # Cálculo paralelo
            derivatives = model.compute_derivatives(states, inputs)
        """
        # Extração de componentes (sem quebrar o grafo computacional)
        # theta está no índice 2, beta no índice 3
        theta = states[:, 2]
        beta = states[:, 3]
        
        v = inputs[:, 0]
        alpha = inputs[:, 1]
        
        # Pré-cálculos trigonométricos
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        sin_beta = torch.sin(beta)
        cos_beta = torch.cos(beta)
        tan_alpha = torch.tan(alpha)
        
        # Construção do vetor de controle transformado u = [v, v*tan(alpha)]
        # Shape: (Batch, 2, 1) para multiplicação matricial
        u_transformed = torch.stack([v, v * tan_alpha], dim=1).unsqueeze(2)
        
        # Construção da Matriz M(q)
        # Shape desejado: (Batch, 4, 2)
        batch_size = states.shape[0]
        zeros = torch.zeros_like(theta)
        
        # Coluna 1 da Matriz (Coeficientes de v)
        # [cos(theta), sin(theta), 0, -1/L * sin(beta)]
        col1 = torch.stack([
            cos_theta,
            sin_theta,
            zeros,
            -self.inv_L * sin_beta
        ], dim=1)
        
        # Coluna 2 da Matriz (Coeficientes de v*tan(alpha))
        # [0, 0, 1/D, k*cos(beta) - 1/D]
        col2 = torch.stack([
            zeros,
            zeros,
            torch.full_like(theta, self.inv_D),
            (self.offset_factor * cos_beta) - self.inv_D
        ], dim=1)
        
        # Empilhamento para formar M(q) -> (Batch, 4, 2)
        M_q = torch.stack([col1, col2], dim=2)
        
        # Multiplicação: (Batch, 4, 2) @ (Batch, 2, 1) -> (Batch, 4, 1)
        q_dot = torch.bmm(M_q, u_transformed).squeeze(2)
        
        return q_dot
