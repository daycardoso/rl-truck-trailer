from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from trailer_gym_env import TrailerDockingEnv
from torch import nn

# 1. Instanciar e Checar Ambiente
env = TrailerDockingEnv(render_mode=None) # Sem render para treinar rápido
check_env(env) # Valida se segue a API do Gym corretamente

# 2. Definir Modelo SAC
# Usamos MlpPolicy pois a entrada é um vetor de 26 valores.
#qi = 128, 64 tanh
#qi = 128, 64 tanh
#qf = 128, 64 tanh

policy_kwargs = dict(
    net_arch=dict(pi=[128, 64], qf=[128, 64]),
    activation_fn=nn.Tanh
)

model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    learning_rate=3e-4,           
    buffer_size=1_000_000,        
    batch_size=512,               
    gamma=0.99,                   
    ent_coef="auto",              
    target_entropy="auto",        
    tau=0.005,                    
    learning_starts=50_000,       
    train_freq=1,                 
    gradient_steps=2,             
    device="auto",
    use_sde=True,                 
    sde_sample_freq=4,  
    tensorboard_log="./tensorboard_logs/"
)

# 3. Callback para salvar modelos periodicamente
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix="sac_trailer"
)

# 4. Treinar
print("Iniciando treinamento...")
model.learn(
    total_timesteps=500_000, 
    callback=checkpoint_callback,
    progress_bar=True
)

# 5. Salvar Final
model.save("sac_trailer_final")
print("Treinamento concluído e modelo salvo.")