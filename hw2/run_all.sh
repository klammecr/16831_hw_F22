# QUESTION 1: Small Scale Experiments
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa -ngpu
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa -ngpu
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na -ngpu
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa -ngpu
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa -ngpu
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na -ngpu

# QUESTION 2: Inverted Pendulum
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.005 -rtg --exp_name q2_b100_r0.005 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.005 -rtg --exp_name q2_b1000_r0.005 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.01 -rtg --exp_name q2_b1000_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.02 -rtg --exp_name q2_b1000_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.025 -rtg --exp_name q2_b1000_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.035 -rtg --exp_name q2_b1000_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.040 -rtg --exp_name q2_b1000_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.050 -rtg --exp_name q2_b1000_r0.050 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.01 -rtg --exp_name q2_b1000_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.02 -rtg --exp_name q2_b1000_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.025 -rtg --exp_name q2_b1000_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.035 -rtg --exp_name q2_b1000_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.040 -rtg --exp_name q2_b1000_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 1000 -lr 0.050 -rtg --exp_name q2_b1000_r0.050 -ngpu
# Batch Size 250
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.005 -rtg --exp_name q2_b250_r0.005 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.01 -rtg --exp_name q2_b250_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.02 -rtg --exp_name q2_b250_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.025 -rtg --exp_name q2_b250_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.035 -rtg --exp_name q2_b250_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.040 -rtg --exp_name q2_b250_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.050 -rtg --exp_name q2_b250_r0.050 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.01 -rtg --exp_name q2_b250_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.02 -rtg --exp_name q2_b250_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.025 -rtg --exp_name q2_b250_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.035 -rtg --exp_name q2_b250_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.040 -rtg --exp_name q2_b250_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 250 -lr 0.050 -rtg --exp_name q2_b250_r0.050 -ngpu
# # Batch Size 100
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.005 -rtg --exp_name q2_b100_r0.005 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.01 -rtg --exp_name q2_b100_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.02 -rtg --exp_name q2_b100_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.025 -rtg --exp_name q2_b100_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.035 -rtg --exp_name q2_b100_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.040 -rtg --exp_name q2_b100_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.050 -rtg --exp_name q2_b100_r0.050 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.01 -rtg --exp_name q2_b100_r0.01 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.02 -rtg --exp_name q2_b100_r0.02 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.025 -rtg --exp_name q2_b100_r0.025 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.035 -rtg --exp_name q2_b100_r0.035 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.040 -rtg --exp_name q2_b100_r0.040 -ngpu
python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000\
 --discount 0.9 -n 100 -l 2 -s 64 -b\
\n 100 -lr 0.050 -rtg --exp_name q2_b100_r0.050 -ngpu

# QUESTION 3: Lunar Lander
python rob831/scripts/run_hw2.py env_name LunarLanderContinuous-v4 --ep_len 1000 discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005 -ngpu

# QUESTION 4: Parameter Search Half Cheetah
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.005_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.005_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.005_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.010 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.010_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.010 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.010_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.010 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.010_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.015 -rtg --nn_baseline --exp_name q4_search_b10000_lr0.015_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.015 -rtg --nn_baseline --exp_name q4_search_b30000_lr0.015_rtg_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.015 -rtg --nn_baseline --exp_name q4_search_b50000_lr0.015_rtg_nnbaseline -ngpu

# QUESTION 4: Final Half Cheetah
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.015 --exp_name q4_b50000_r0.015 -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.015 -rtg --exp_name q4_b50000_r0.015_rtg -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.015 --nn_baseline --exp_name q4_b50000_r0.015_nnbaseline -ngpu
python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.015 -rtg --nn_baseline --exp_name q4_b50000_r0.015_rtg_nnbaseline -ngpu

# QUESTION 5: GAE Hopper v4
python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 \
-l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5\
 --gae_lambda 0 --exp_name q5_b2000_r0.001_lambda0 -ngpu
python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 \
-l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5\
 --gae_lambda 0.95 --exp_name q5_b2000_r0.001_lambda0.95 -ngpu
python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 \
-l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5\
 --gae_lambda 0.99 --exp_name q5_b2000_r0.001_lambda0.99 -ngpu
python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 \
-l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5\
 --gae_lambda 1.0 --exp_name q5_b2000_r0.001_lambda1.0 -ngpu