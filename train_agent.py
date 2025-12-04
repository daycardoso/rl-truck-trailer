from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from trailer_gym_env import TrailerDockingEnv
from torch import nn

if __name__ == "__main__":
    # 1. Instanciar e Checar Ambiente (Opcional, apenas para validação inicial)
    # check_env(TrailerDockingEnv(render_mode=None)) 

    # 2. Configurar Ambientes Paralelos
    # SubprocVecEnv é essencial para paralelismo real em Python (multiprocessing)
    n_envs = 16  # Defina com base no seu CPU (ex: número de threads - 2)
    env = make_vec_env(TrailerDockingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # 3. Definir Modelo SAC
    # Usamos MlpPolicy pois a entrada é um vetor de 26 valores.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 32], qf=[128, 32]),
        activation_fn=nn.Tanh
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,                    # Aumentei verbosidade para ver progresso dos envs
        learning_rate=3e-4,           
        buffer_size=1_000_000,        
        batch_size=512,               
        gamma=0.99,                   
        ent_coef="auto",              
        target_entropy="auto",        
        tau=0.005,                    
        learning_starts=50_000,       
        train_freq=1,                 
        gradient_steps=1,             
        device="auto",
        use_sde=True,                 
        sde_sample_freq=4,  
        tensorboard_log="./tensorboard_logs/"
    )

    # 4. Callback para salvar modelos periodicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs, # Ajusta frequência pois steps são contados por env
        save_path="./logs/",
        name_prefix="sac_trailer"
    )

    # 5. Treinar
    print(f"Iniciando treinamento com {n_envs} ambientes paralelos...")
    model.learn(
        total_timesteps=20000000, 
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 6. Salvar Final
    model.save("sac_trailer_final")
    print("Treinamento concluído e modelo salvo.")
    env.close()