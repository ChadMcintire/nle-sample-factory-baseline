import os
import sys
import time
from collections import deque
from utils.graphs import GraphBuilder
import numpy as np
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict

# Needs to be imported to register models and envs
import models as my_models
import env as my_envs

# Benji's log file/class
from log import ActionLog

from collections import OrderedDict
import time


# TARGET_NUM_EPISODES = 512
TARGET_NUM_EPISODES = 50

def enjoy(cfg, max_num_frames=1e9, target_num_episodes=TARGET_NUM_EPISODES):
    """
    This is a modified version of original appo.enjoy_appo.enjoy function,
    modified to have an episode limit.
    """
    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = 1
    cfg.num_envs = 1
    cfg.use_aicrowd_gym = True

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    #added the obsevation space to the end, the "tty cursor" object is just to render the image, and can be removed 
    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    # sample-factory defaults to work with multiagent environments,
    # but we can wrap a single-agent env into one of these like this
    env = MultiAgentWrapper(env)

    log.info("Num agents {}".format(env.num_agents))

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_max_depth = []
    episode_max_turn = []
    episodic_action_attempts = []
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0
    num_episodes = 0

    obs = env.reset()
    origin_obs = obs
    #print(env.env)
    #print(obs)
    #print(obs[0].keys())
    #print(obs[0]["obs"])
    #print(obs[0]["vector_obs"])
    #print(len(obs[0]["vector_obs"]))
    #print("\n\n\n\n")
    #print(env.env.blstats)
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    # create instance of log to keep track of agent actions
    action_log = ActionLog(env)

    # create graph directory and builder for heatmap
    g_type = "heat_pos"
    builder = GraphBuilder([g_type])


    new_dict_comp = {n:0 for n in list(range(env.action_space.n))}
    message_dict = OrderedDict()
    max_depth = 0
    turn_of_arrival_of_max_depth = 0
    attempted_actions = 0

    with torch.no_grad():
        while num_frames < max_num_frames and num_episodes < target_num_episodes:
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

            # sample actions from the distribution by default
            actions = policy_outputs.actions

            # log worker action
            action_log.record_action(actions.item())

            actions = actions.cpu().numpy()
            
            #get actions for graphic etc.
            new_dict_comp[policy_outputs.actions.item()] += 1

            rnn_states = policy_outputs.rnn_states

            turn_of_arrival_of_max_depth = env.env.blstats[20] 
            #print("env val", env.env.blstats[24])
            #print("turn_val", env.env.blstats[20])
            if max_depth < env.env.blstats[24]:
                #seems to be the right amount of time to wait for the parent process to catch up to the env for the
                #blstats to be correct
                time.sleep(5)
                max_depth = env.env.blstats[24] 
                print("env val2", env.env.blstats[24])
                print("turn_val2", env.env.blstats[20])
                #max_depth = env.env.blstats[24] 
                #turn_of_arrival_of_max_depth = env.env.blstats[20] 
                print("max_depth", max_depth)
                print("turns", turn_of_arrival_of_max_depth)
                #time.sleep(10)

                # generate heat maps and save to {WorkingDir}/graphs/{g_type}
                #print("blstats", env.env.blstats)
                #loc = os.getcwd() + "/graphs/" 
                #if not os.path.exists(loc):
                #    os.makedirs(loc)
                #builder.save_graphs(loc, env.env.blstats[24])
                ##time.sleep(5)
                ##builder.set_data(g_type, [])
                #builder = GraphBuilder([g_type])

                #sys.exit()

            obs, rew, done, infos = env.step(actions)

            message = "".join([chr(n) for n in obs[0]["message"] if chr(n) != "\x00"])
            if message in message_dict.keys():
                message_dict[message] += 1
            else:
                message_dict[message] = 1


            # add positions to heatmap
            #x, y, *other = env.env.blstats

            #builder.append_point(g_type, (x, y))

            # render the game
            #env.render()

            episode_reward += rew
            num_frames += 1
            attempted_actions += 1 
            #print(num_frames)

            for agent_i, done_flag in enumerate(done):
                if done_flag:
                    #print("env done", env.env.blstats[24])
                    #print("turn_done", env.env.blstats[20])
                    #time.sleep(3)
                    # print actions after episode
                    action_log.print_actions()
                    # clear actions for next episode
                    action_log.clear_actions()
                    
                    finished_episode[agent_i] = True
                    episode_rewards[agent_i].append(episode_reward[agent_i])
                    true_rewards[agent_i].append(infos[agent_i].get('true_reward', episode_reward[agent_i]))
                    log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f', agent_i, num_frames, episode_reward[agent_i], true_rewards[agent_i][-1])
                    rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                    episode_reward[agent_i] = 0

                    #need to make this multi-agent later
                    print("\n\n\n", max_depth)
                    print("\n\n\n", turn_of_arrival_of_max_depth)
                    episode_max_depth.append(max_depth)
                    episode_max_turn.append(turn_of_arrival_of_max_depth)
                    episodic_action_attempts.append(attempted_actions)

                    max_depth = 0
                    turn_of_arrival_of_max_depth = 0
                    attempted_actions = 0
                    num_episodes += 1

            if all(finished_episode):
                finished_episode = [False] * env.num_agents
                avg_episode_rewards_str, avg_true_reward_str = '', ''
                for agent_i in range(env.num_agents):
                    avg_rew = np.mean(episode_rewards[agent_i])
                    avg_true_rew = np.mean(true_rewards[agent_i])
                    if not np.isnan(avg_rew):
                        if avg_episode_rewards_str:
                            avg_episode_rewards_str += ', '
                        avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                    if not np.isnan(avg_true_rew):
                        if avg_true_reward_str:
                            avg_true_reward_str += ', '
                        avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'

                log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_reward_str)
                log.info('Avg episode reward: %.3f, avg true_reward: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_rewards[i]) for i in range(env.num_agents)]))

    #print(new_dict_comp)
    #for key, value in message_dict.items():
    #    print(key, value)
    print("max_depth", max_depth)
    print("turn count", turn_of_arrival_of_max_depth)
    print("episode turn", episode_max_turn)
    print("episode depth", episode_max_depth)
    avg_max_turn = np.mean(episode_max_turn)
    avg_max_depth = np.mean(episode_max_depth)
    min_episodic_turn = np.min(episode_max_turn)
    min_episodic_depth = np.min(episode_max_depth)
    max_episodic_turn = np.max(episode_max_turn)
    max_episodic_depth = np.max(episode_max_depth)
    average_attempted_action = np.mean(episodic_action_attempts)
    max_attempted_action = np.max(episodic_action_attempts)


    print("average turn", avg_max_turn)
    print("average depth",avg_max_depth)
    print("min episodic turn", min_episodic_turn)
    print("min episodic depth", min_episodic_depth )
    print("max episodic turn", max_episodic_turn)
    print("max episodic depth", max_episodic_depth )
    print("average attempted actions", average_attempted_action)
    print("max attempted actions", max_attempted_action)

    # generate heat maps and save to {WorkingDir}/graphs/{g_type}
    #loc = os.getcwd() + "/graphs/"
    #if not os.path.exists(loc):
    #    os.makedirs(loc)
    #builder.save_graphs(loc, max_depth)
    #builder.set_data(g_type, [])

    list_of_elem = [avg_max_turn, avg_max_depth, min_episodic_turn, min_episodic_depth, max_episodic_turn, max_episodic_depth, average_attempted_action, max_attempted_action]
    from csv import writer
    with open("Data_Collection/Episodic_Returns.csv", 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Evaluation entry point."""
    cfg = parse_all_args()
    #print("\n\n\ncfg",cfg, "\n\n\n")
    _ = enjoy(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
