import torch

class VehicleGeometryGPU:
    """
    Calcula os vértices do Trator e do Trailer para visualização e detecção de colisão.
    
    Implementação vetorizada (PyTorch) baseada nas equações geométricas fornecidas.
    Suporta processamento em lote (Batch Processing).
    
    Parâmetros Geométricos (baseados no snippet):
    --------------------------------------------
    width_tractor : Largura total do trator.
    width_trailer : Largura total do trailer.
    length_tractor : Comprimento total do trator.
    length_trailer : Comprimento total do trailer.
    d_axle_kingpin : Distância eixo traseiro trator -> Quinta roda (M).
    d_front_kingpin : Distância frente trator -> Quinta roda (para cálculo da traseira).
    d_front_trailer_kingpin : Distância frente trailer -> Pino rei.
    """

    def __init__(self, config: dict):
        # Extração de parâmetros com valores default (em metros)
        self.w_tractor = config.get('width_tractor', 2.5)
        self.w_trailer = config.get('width_trailer', 2.5)
        self.l_tractor = config.get('length_tractor', 5.0)
        self.l_trailer = config.get('length_trailer', 13.6) # Padrão carreta longa
        
        # M: Distância entre o eixo traseiro do trator e a quinta roda
        self.d_axle_kingpin = config.get('d_axle_kingpin', 0.5) 
        
        # Parâmetros de balanço (Overhang)
        self.d_front_kingpin = config.get('d_front_kingpin', 3.5)
        self.d_front_trailer_kingpin = config.get('d_front_trailer_kingpin', 1.5)
        
        # Pré-cálculo de meias-larguras
        self.half_w_tractor = self.w_tractor / 2.0
        self.half_w_trailer = self.w_trailer / 2.0

    def compute_vertices(self, states: torch.Tensor) -> dict:
        """
        Gera os vértices para um lote de estados.

        Parâmetros
        ----------
        states : torch.Tensor
            Tensor (Batch, 4) -> [x, y, theta_trator, beta]

        Retorno
        -------
        dict
            {
                'tractor': torch.Tensor (Batch, 4, 2),
                'trailer': torch.Tensor (Batch, 4, 2)
            }
            Ordem: Traseira-Esq, Traseira-Dir, Dianteira-Dir, Dianteira-Esq
        """
        # 1. Desempacotar Estados
        # x1, y1, theta1
        x_trator = states[:, 0]
        y_trator = states[:, 1]
        ang_trator = states[:, 2]
        beta = states[:, 3]
        
        # theta2 = theta1 + beta
        ang_trailer = ang_trator + beta

        # 2. Pré-cálculo Trigonométrico (Batch)
        cos_t1 = torch.cos(ang_trator)
        sin_t1 = torch.sin(ang_trator)
        cos_t2 = torch.cos(ang_trailer)
        sin_t2 = torch.sin(ang_trailer)

        # 3. Quinta Roda / Pino Rei (Q)
        # Q = (x1 + M cos(theta1), y1 + M sen(theta1))
        qx = x_trator + self.d_axle_kingpin * cos_t1
        qy = y_trator + self.d_axle_kingpin * sin_t1

        # =========================================================
        # TRATOR (Cálculos relativos ao Eixo Traseiro e Quinta Roda)
        # =========================================================
        
        # Auxiliares de deslocamento (seguindo estritamente seu snippet)
        # Frente Trator
        dx_long_front_trator = self.l_tractor * cos_t1
        dy_long_front_trator = self.l_tractor * sin_t1
        dx_lat_trator = self.half_w_tractor * sin_t1
        dy_lat_trator = self.half_w_tractor * cos_t1
        
        # Traseira Trator (Calculada a partir da Quinta Roda conforme snippet)
        # (Ct_trator - D_FrTrator)
        dist_traseira = self.l_tractor - self.d_front_kingpin
        dx_long_rear_trator = dist_traseira * cos_t1
        dy_long_rear_trator = dist_traseira * sin_t1

        # Vértices Trator
        # V_FD (Frente Direita)
        v_fd_x = x_trator + dx_long_front_trator + dx_lat_trator
        v_fd_y = y_trator + dy_long_front_trator - dy_lat_trator
        
        # V_FE (Frente Esquerda)
        v_fe_x = x_trator + dx_long_front_trator - dx_lat_trator
        v_fe_y = y_trator + dy_long_front_trator + dy_lat_trator
        
        # V_TD (Traseira Direita) - relativo a Q
        v_td_x = qx - dx_long_rear_trator + dx_lat_trator
        v_td_y = qy - dy_long_rear_trator - dy_lat_trator
        
        # V_TE (Traseira Esquerda) - relativo a Q
        v_te_x = qx - dx_long_rear_trator - dx_lat_trator
        v_te_y = qy - dy_long_rear_trator + dy_lat_trator

        # Empilhar vértices do Trator: [TE, TD, FD, FE]
        # Shape final: (Batch, 4, 2)
        tractor_verts = torch.stack([
            torch.stack([v_te_x, v_te_y], dim=1),
            torch.stack([v_td_x, v_td_y], dim=1),
            torch.stack([v_fd_x, v_fd_y], dim=1),
            torch.stack([v_fe_x, v_fe_y], dim=1)
        ], dim=1)

        # =========================================================
        # TRAILER (Cálculos relativos ao Pino Rei Q)
        # =========================================================
        
        # Auxiliares Trailer
        dx_long_front_trailer = self.d_front_trailer_kingpin * cos_t2
        dy_long_front_trailer = self.d_front_trailer_kingpin * sin_t2
        dx_lat_trailer = self.half_w_trailer * sin_t2
        dy_lat_trailer = self.half_w_trailer * cos_t2
        
        # Distância da Quinta Roda até a traseira do trailer
        dist_traseira_trailer = self.l_trailer - self.d_front_trailer_kingpin
        dx_long_rear_trailer = dist_traseira_trailer * cos_t2
        dy_long_rear_trailer = dist_traseira_trailer * sin_t2

        # Vértices Trailer
        # V_FD (Frente Direita) - Relativo a Q
        t_fd_x = qx + dx_long_front_trailer + dx_lat_trailer
        t_fd_y = qy + dy_long_front_trailer - dy_lat_trailer
        
        # V_FE (Frente Esquerda) - Relativo a Q
        t_fe_x = qx + dx_long_front_trailer - dx_lat_trailer
        t_fe_y = qy + dy_long_front_trailer + dy_lat_trailer
        
        # V_TD (Traseira Direita) - Relativo a Q
        t_td_x = qx - dx_long_rear_trailer + dx_lat_trailer
        t_td_y = qy - dy_long_rear_trailer - dy_lat_trailer
        
        # V_TE (Traseira Esquerda) - Relativo a Q
        t_te_x = qx - dx_long_rear_trailer - dx_lat_trailer
        t_te_y = qy - dy_long_rear_trailer + dy_lat_trailer

        # Empilhar vértices do Trailer: [TE, TD, FD, FE]
        trailer_verts = torch.stack([
            torch.stack([t_te_x, t_te_y], dim=1),
            torch.stack([t_td_x, t_td_y], dim=1),
            torch.stack([t_fd_x, t_fd_y], dim=1),
            torch.stack([t_fe_x, t_fe_y], dim=1)
        ], dim=1)

        return {
            'trator': tractor_verts,
            'trailer': trailer_verts
        }