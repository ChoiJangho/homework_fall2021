Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

### Problem 1.2

**Ant environment:**

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 --eval_bacth_size 10000 --num_agent_train_steps_per_iter 2000
```

**Humanoid environment:**

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 --eval_bacth_size 100000 --num_agent_train_steps_per_iter 2000
```

### Problem 1.3
Same parameters for each environment, with `num_agent_train_steps_per_iter` varying from 400, 800, 1200, 1600, 2000.

### Problem 2.1

**Ant environment:**

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 --eval_bacth_size 10000 --num_agent_train_steps_per_iter 2000
```

**Humanoid environment:**

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 --eval_bacth_size 100000 --num_agent_train_steps_per_iter 1200
```
