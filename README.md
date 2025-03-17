# What's in this repo

1. Codebases for H1 rl training  
2. Some retargeting methods. 

# Codebases for H1 rl training
We implement two different h1 rl environments (humanplus and jaylon's) by using different config documents under the same h1.py.
### Characteristics of humanplus 
1. Full h1 dofs (19 dofs)  
2. Same stiffness and damping for all joints (100/5)  
3. ActorCriticTransformer  
4. Lower learning rate
   
Train with humanplus environment:

    python legged_gym/scripts/train.py --run_name 0001_test --headless --sim_device cuda:0 --rl_device cuda:0

### Characteristics of jaylon's 
1. Imcomplete h1 dofs (10 dofs, only lower body)  
2. Different stiffness and damping for different joints   
3. Simple MLP ActorCritic network
4. Higher learning rate (together with 3 makes it faster to train)
   
Train with jaylon's environment:

    python legged_gym/scripts/train.py --run_name 0001_test --headless --sim_device cuda:0 --rl_device cuda:0 --task h1_jaylon

### Why different environment?
As mentioned before, there is several major difference make them different environments. Besides, there is also minor differences between them (e.g., only_positive_rewards, action_scale...). All these factors together influence the final results of a trained policy.
