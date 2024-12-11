import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import environment_beam_multiple as environment_beam
import env_setups_vector as env_setups
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


UIDS = ['1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582', '1.3.6.1.4.1.14519.5.2.1.1706.8040.222028698245669229227096139476', '1.3.6.1.4.1.14519.5.2.1.1706.8040.289247325326235527684687492278', '1.3.6.1.4.1.22213.2.26556']

def decrease_learning_rate(lr):
    if lr > 0.001 * (0.9**5) : 
        return lr * 0.9
    else:
        lr = 0.001
        return lr

def reset_learning_rate(lr):
    lr = 0.003
    return lr

def stationary_rate(lr):
    return lr

class LearningRateScheduler(BaseCallback):
    def __init__(self, learning_rate_func, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.lr_schedule_fn = learning_rate_func

    def _on_step(self):

        current_lr = self.model.lr_schedule(1)
        # new_lr = self.lr_schedule_fn(current_lr)
        self.model.lr_schedule = lambda _, initial_lr=current_lr: initial_lr
        
        return True
    
    
    def _on_training_start(self):
        """
        This event is triggered before exiting the `learn()` method.
        """
        current_lr = self.model.lr_schedule(1)
        new_lr = self.lr_schedule_fn(current_lr)
        self.model.lr_schedule = lambda _, initial_lr=new_lr: initial_lr
        return True

def make_env(env_params, logdir, uid , slice_index):
        def _init():
            env = environment_beam.Beam_Env(*env_params)
            env = Monitor(env, logdir +'/'+ str(slice_index) + '  '+ str(uid))
            return env
        return _init


def load_dataset(slice_nums):
    env_param_list = []
    print(slice_nums)
    for slice_num, uid in slice_nums:
        env_setup = env_setups.Beam_Env(uid,slice_num)
        env_params = env_setup.get_params()
        env_param_list.append(env_params)
    return env_param_list

def update_environment(env, new_slice_nums, logdir):
    env.close()  # Close the current environments
    new_env_param_list = load_dataset(new_slice_nums)
    #new_env_list = [make_env(_env_parameters, logdir, data[1], data[0]) for _env_parameters, data in zip(new_env_param_list, new_slice_nums)]
    #new_env = SubprocVecEnv(new_env_list)
    new_env = environment_beam.Beam_Env(*new_env_param_list[0])
    return new_env

def main(LOOP):

    learning_rates = [0.001, 0.0001, 0.01]
    discount_factors = [0.99, 0.95, 0.9]
    entropy_coefficients = [0.01, 0.05, 0.1]
    clip_ranges = [0.3, 0.2, 0.4]

    learning_rates = [0.001]
    discount_factors = [0.99]
    entropy_coefficients = [0.01]
    clip_ranges = [0.3]

    initial_slice_nums = [53,93]
    initial_uids= [UIDS[0],UIDS[2]]

    initial_slice_nums = [58,60]
    initial_uids= [UIDS[3],UIDS[3]]

    initial_slice_nums = [58]
    initial_uids= [UIDS[3]]

    initial_data_set = np.column_stack((initial_slice_nums,initial_uids))


    next_data_sets = {}

    slice_num_array = [[[UIDS[3],UIDS[2]], [76,80]],
        [[UIDS[0], UIDS[3]], [42,68]],
        [[UIDS[0],UIDS[1]],[36,80]],
        [[UIDS[3],UIDS[0]],[63,30]],
        [[UIDS[1],UIDS[3]],[75,71]]  ]
    
    slice_num_array = [[[UIDS[3],UIDS[3]], [62,64]],
                       [[UIDS[3],UIDS[3]], [66,67]],
                       [[UIDS[3],UIDS[3]], [55,53]],
                       [[UIDS[3],UIDS[3]], [58,60]],
                       ]
    
    slice_num_array = [[[UIDS[3]], [62]],
                       [[UIDS[3]], [66]],
                       [[UIDS[3]], [55]],
                       [[UIDS[3]], [58]],
                       ]
    

    i=1
    for next_uid, next_slice_nums in slice_num_array:
        # next_uid = '1.3.6.1.4.1.14519.5.2.1.1706.8040.141473127645431244275904143582'
        next_data_sets[i] = np.column_stack((next_slice_nums,next_uid))
        i += 1

    

    TIMESTEPS = 20000

    for alpha in learning_rates:
        for gamma in discount_factors:
            for beta in entropy_coefficients:
                for epsilon in clip_ranges:

    # alpha = learning_rates[0]
    # gamma = discount_factors[0]
    # beta = entropy_coefficients[0]
    # epsilon = clip_ranges[0]

                    dir = f"results/diff_obs_space_125/{LOOP}_obs_2_binary_alpa_{alpha}_gamma_{gamma}_beta_{beta}_eps_{epsilon}_125_steps"
                    models_dir = f"{dir}/models"
                    logdir = f"{dir}/logs"


                    if not os.path.exists(dir): os.makedirs(dir)
                    if not os.path.exists(models_dir): os.makedirs(models_dir)
                    if not os.path.exists(logdir): os.makedirs(logdir)

                    env_param_list = load_dataset(initial_data_set)
                    #env_list = [make_env(_env_parameters, logdir, data[1], data[0]) for _env_parameters, data in zip(env_param_list, initial_data_set)]
                    #env = SubprocVecEnv(env_list)
                    env = environment_beam.Beam_Env(*env_param_list[0])

                    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda:1',
                                learning_rate=alpha,
                                gamma=gamma,
                                ent_coef=beta,
                                clip_range=epsilon,
                                )

                    loop_index = len(slice_num_array)
                    
                    # loop_index = 5
                    j=1
                    k=1


                    NUM_EPS = 200
                    for i in range(NUM_EPS):

                        if i in np.arange(loop_index,NUM_EPS,loop_index):

                            if j == loop_index + 1:
                                j = 1
                                k += 1

                            if k%2 == 1:
                                lr_scheduler = LearningRateScheduler(learning_rate_func=decrease_learning_rate)

                                next_data_set = next_data_sets[j]
                                env = update_environment(env, next_data_set, logdir)  # Transition to new dataset

                                j += 1
                                
                                model.set_env(env)

                                # if i in np.arange(loop_index*3,100,loop_index*3):
                                #     lr_scheduler = LearningRateScheduler(learning_rate_func=reset_learning_rate)
                                # else:
                                #     lr_scheduler = LearningRateScheduler(learning_rate_func=decrease_learning_rate)
                                    
                                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="logs", callback=lr_scheduler)

                                model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")
                            
                            else:
                                next_data_set = next_data_sets[j]
                                env = update_environment(env, next_data_set, logdir)  # Transition to new dataset

                                j += 1
                                
                                model.set_env(env)

                                # if i in np.arange(loop_index*3,100,loop_index*3):
                                #     lr_scheduler = LearningRateScheduler(learning_rate_func=reset_learning_rate)
                                # else:
                                #     lr_scheduler = LearningRateScheduler(learning_rate_func=decrease_learning_rate)
                                    
                                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="logs")
                                if i in np.arange(1,TIMESTEPS,4):

                                    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")

                        else:
                            # model_fixed_lr.set_env(env)  # Important to update the model's environment
                            model.set_env(env)

                            # model_fixed_lr.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Fixed_lr")
                            # lr_scheduler = LearningRateScheduler(learning_rate_func=decrease_learning_rate)
                            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="logs")

                            # model_fixed_lr.save(f"{models_dir}/fixed_lr/{TIMESTEPS*(i+1)}")
                            model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")

    env.close()

if __name__ == '__main__':
    main(1)
    main(2)
    main(3)
    main(4)









































def pretrain(): 

    actions= [(50,50,3),(51,51,3), (51,51,2)]
    env_setup = env_setups.Beam_Env(UIDS[0],'58')
    env_params = env_setup.get_params()

    env = environment_beam.Beam_Env(*env_params)

    observations = []
    rewards = []
    done = False
    observation = env.reset()

    for i in range(len(actions)):
    # Choose an action from your list
        action = actions[len(observations)]

        # Take a step in the environment
        observation, reward, done, _ ,_= env.step(action)
        

        # Record the observation and reward
        observations.append(observation)
        rewards.append(reward)

    # Step 2: Create Expert Demonstrations
    print(actions)
    print(observations)
    expert_demonstrations = [(obs, action) for obs, action in zip(observations, actions)]

    # Step 3: Pre-Train the Model
    # Preprocess the expert demonstrations dataset
    expert_states, expert_actions = zip(*expert_demonstrations)
    expert_states = np.array(expert_states)
    expert_actions = np.array(expert_actions)

    print(expert_actions.shape)
    print(expert_states.shape)

    # Create and train a supervised learning model using expert demonstrations
    # For example, using scikit-learn, TensorFlow, or PyTorch
    # Here, I'll use scikit-learn as an example
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(expert_states, expert_actions)

    # Step 4: Fine-Tune with Reinforcement Learning
    # Create a PPO model with the same policy as the pre-trained model
    pretrained_model = PPO("MlpPolicy", env, verbose=1)

    # Function to convert state to action using the pre-trained model
    def get_action(observation):
        # Use the pre-trained model to predict the action
        action_prob = model.predict_proba([observation])[0]
        action = np.random.choice(len(action_prob), p=action_prob)
        return action

    # Train the pre-trained model using expert demonstrations
    pretrained_model.learn(total_timesteps=10000, expert_demos=expert_demonstrations, get_action=get_action)



# pretrain()

