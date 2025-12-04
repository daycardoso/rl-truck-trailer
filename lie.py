import sympy as sp
import numpy as np

def derive_privileged_coordinates():
    """
    Deriva as coordenadas privilegiadas (z) para o modelo Trator-Trailer
    usando o Design Procedure de Rosenfelder (Etapa 2 e 3).
    """
    # 1. Variáveis Simbólicas
    x, y, theta, beta = sp.symbols('x y theta beta')
    v, u_alpha = sp.symbols('v u_alpha') # u_alpha = v * tan(alpha)
    inv_D, inv_L, k = sp.symbols('inv_D inv_L k')
    
    state = sp.Matrix([x, y, theta, beta])
    
    # 2. Definição dos Campos Vetoriais (Dynamics)
    # g1 acompanha v
    g1 = sp.Matrix([
        sp.cos(theta),
        sp.sin(theta),
        0,
        -inv_L * sp.sin(beta)
    ])
    
    # g2 acompanha u_alpha (entrada de esterçamento)
    g2 = sp.Matrix([
        0,
        0,
        inv_D,
        k * sp.cos(beta) - inv_D
    ])
    
    # 3. Cálculo dos Colchetes de Lie na Origem (q=0)
    # Função auxiliar para Colchete de Lie: [f, g] = jac(g)*f - jac(f)*g
    def lie_bracket(f, g, q):
        jac_g = g.jacobian(q)
        jac_f = f.jacobian(q)
        return (jac_g * f) - (jac_f * g)

    # Avaliar g1 e g2 na origem
    zeros = {x:0, y:0, theta:0, beta:0}
    X1 = g1.subs(zeros)
    X2 = g2.subs(zeros)
    
    # X3 = [g1, g2]
    lg1g2 = lie_bracket(g1, g2, state)
    X3 = lg1g2.subs(zeros)
    
    # X4 = [g1, [g1, g2]] -> Vetor de crescimento (2, 3, 4)
    lg1g1g2 = lie_bracket(g1, lg1g2, state)
    X4 = lg1g1g2.subs(zeros)
    
    # 4. Montagem da Matriz A (Adapted Frame)
    A = sp.Matrix.hstack(X1, X2, X3, X4)
    
    print("Matriz A (Adapted Frame):")
    sp.pprint(A)
    
    # 5. Cálculo de z = A^(-1) * q
    # Verifica se é inversível (Controlabilidade)
    if A.det() == 0:
        print("\nErro: Sistema não é linearmente controlável na origem com estes colchetes.")
        return

    A_inv = A.inv()
    z = A_inv * state
    
    print("\nCoordenadas Privilegiadas (z):")
    for i in range(4):
        print(f"z_{i+1} = {sp.simplify(z[i])}")

    return z

# Execute para ver as equações
derive_privileged_coordinates()