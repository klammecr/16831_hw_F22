#python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1
#python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2
#python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3

# python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1
# python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2
# python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3

# Parameter Tuning!
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --hidden_layers 2 --hidden_units 32 --exp_name q3_hparam1
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --hidden_layers 2 --hidden_units 64 --exp_name q3_hparam2
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --hidden_layers 2 --hidden_units 128 --exp_name q3_hparam3
python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --hidden_layers 2 --hidden_units 256 --exp_name q3_hparam4

# Evaluation Question 4
# python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
# python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_100_1 -ntu 100 -ngsptu 1
# python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_1_100 -ntu 1 -ngsptu 100
# python rob831/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_10_10 -ntu 10 -ngsptu 10

# Evaluation Question 5
# python rob831/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_1_100 -ntu 1 -ngsptu 100
# python rob831/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_1_100 -ntu 1 -ngsptu 100
