import environment_beam_multiple as environment_beam
#import environment_beam as environment_beam
# import env_setups_vector as env_setups
import env_setups_vector as env_setups
from stable_baselines3 import PPO

UIDS = ['1.3.6.1.4.1.22213.2.26556','1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582']

env_setup = env_setups.Beam_Env(UIDS[0],'58')
env_params = env_setup.get_params()

env = environment_beam.Beam_Env(*env_params)

models_dir = "/MPhysGamification/Semester_2/vector_test/results/obsverve_single_env/alpa_0.01_gamma_0.95_beta_0.05_eps_0.2_200_steps/models"
#models_dir= "results/incorrecty_x_axis_for_boundary/models_fixed_rewards_indices_diff_obs"
#models_dir = "vector_env_ok/models_h_2"

env.reset()
model_path = f"{models_dir}/200000"
model= PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    # if ep == 1:
    #     env.debug(DOSE=False, TARGET=False, SAVE=True)    
    obs, info = env.reset()
    done = False
    reward_total = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        reward_total += reward
    env.render(circles=False, show_fig=False, save_fig=True, RESULTS=True, DOSE=True,fig_name=f"{ep}.png")
    env.beam_number()
    print("Episode number: ", ep + 1, ". Reward: ", reward_total)

env.close()