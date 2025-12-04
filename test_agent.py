from stable_baselines3 import SAC
from trailer_gym_env import TrailerDockingEnv

# Carregar ambiente com renderização humana
env = TrailerDockingEnv(render_mode="human")
model = SAC.load("logs\\sac_trailer_60000_steps.zip")

obs, _ = env.reset()
done = False

print("Iniciando Teste Visual...")
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()