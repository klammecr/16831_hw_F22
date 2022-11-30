# Q1 Experiments
python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_medium_dqn

python rob831/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_medium_cql --exploit_rew_shift 1 --exploit_rew_scale 100

python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=0.0 --offline_exploitation \
--exp_name q1_hard_dqn

python rob831/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
--num_exploration_steps=20000 --cql_alpha=1.0 --offline_exploitation \
--exp_name q1_hard_cql --exploit_rew_shift 1 --exploit_rew_scale 100


lambda=(0.1 1 2 10 20 50)
for (( i =  0; i < 6; ++i))
do
    # Q2 RWR
    python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0
    --exp_name q2_rwr_medium_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]}--num_exploration_steps=20000 --rwr

    python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0
    --exp_name q2_rwr_easy_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]} --num_exploration_steps=20000 --rwr

    # Q3 AWR
    python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0
    --exp_name q3_awr_medium_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]} --num_exploration_steps=20000 --awr
    
    python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0
    --exp_name q3_awr_easy_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]} --num_exploration_steps=20000 --awr

    # Q4 AWAC
    python rob831/scripts/run_hw5_awac.py --env_name PointmassEasy-v0
    --exp_name q4_awac_easy_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]} --num_exploration_steps=20000

    python rob831/scripts/run_hw5_awac.py --env_name PointmassMedium-v0
    --exp_name q4_awac_medium_lam${lambda[$i]} --use_rnd
    --offline_exploitation --awac_lambda=${lambda[$i]} --num_exploration_steps=20000
done