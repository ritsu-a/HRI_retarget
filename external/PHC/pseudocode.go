// struct
self IMAmpAgent(AMPAgent(CommonAgent(A2CAgent)))

    self.model ModelAMPContinuous.Network

        self.a2c_network AMPBuilder.Network
            
    self.experience_buffer(['obses', 'rewards', 'values', 'neglogpacs', 'dones', 'actions', 'mus', 'sigmas', 'next_obses', 'next_values', 'amp_obs'])
    self._amp_obs_demo_buffer
    self._amp_replay_buffer


// train pipeline
CommonAgent.train:
    self.obs = self.env_reset()
        obs = self.vec_env.reset(env_ids): torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            vec_env._reset_envs(env_ids)
            Humanoid._reset_envs(env_ids):
                self._reset_actors(env_ids) # this funciton calle _set_env_state, and should set all state vectors 
                    HumanoidAMP._reset_ref_state_init:
                        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._sample_ref_state(env_ids)
                            MotionLibBase.get_motion_state(motion_ids, motion_times, offset=offset):
                                local_rot = self.lrs <- motion.local_rotation <- curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
                                dof_pos = self._local_rotation_to_dof_smpl(local_rot)
                        self._set_env_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos, root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel, rigid_body_pos=rb_pos, rigid_body_rot=rb_rot, rigid_body_vel=body_vel, rigid_body_ang_vel=body_ang_vel)

                self._reset_env_tensors(env_ids):
                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
                    self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
             


        obs = self.obs_to_tensors(obs)

    AMPAgent.train_epoch:
        play_steps: // no_grad
            -> batch_dict(['actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'obses', 'dones', 'next_obses', 'amp_obs', 'returns', 'terminated_flags', 'reward_raw', 'played_frames', 'disc_rewards', 'mb_rewards'])
                
            for (self.horizon_length):
                res_dict = self.get_action_values(self.obs)
                    self.model(input_dict)
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                next_vals = self._eval_critic(self.obs)
                    c_out = self.critic_cnn(obs)
                    c_out = self.critic_mlp(c_out)
                    -> value = self.value_act(self.value(c_out))
                self.experience_buffer.update()

                self._calc_amp_rewards(mb_amp_obs)
                    -> disc_r = -torch.log(1 - self._eval_disc(amp_obs)) // prob -> 1
                mb_advs = self.discount_values(self.dones, self.experience_buffer['values'], rewards, next_vals)
                mb_returns = mb_advs + self.experience_buffer['values']

        batch_dict['amp_obs_demo'] = self._amp_obs_demo_buffer.sample() // store from data
        batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample() // store from play_steps

        for (mini_epochs_num):
            for data in dataset:
                input_dict(['old_values', 'old_logp_actions', 'advantages', 'returns', 'actions', 'obs', 'mu', 'sigma', 'amp_obs', 'amp_obs_demo', 'amp_obs_replay'])

                train_actor_critic: AMPAgent.calc_gradients:
                    -> curr_train_info(['entropy', 'kl', 'last_lr', 'lr_mul', 'b_loss', 'actor_loss', 'actor_clipped', 'actor_clip_frac', 'critic_loss', 'disc_loss', 'disc_grad_penalty', 'disc_logit_loss', 'disc_agent_acc', 'disc_demo_acc', 'disc_agent_logit', 'disc_demo_logit'])
                    
                    self.model(input_dict)
                        input_dict(['is_train', 'amp_steps', 'prev_actions', 'obs', 'amp_obs', 'amp_obs_replay', 'amp_obs_demo', 'obs_orig'])
                        -> res_dict(['prev_neglogp', 'values', 'entropy', 'rnn_states', 'mus', 'sigmas', 'disc_agent_logit', 'disc_agent_replay_logit', 'disc_demo_logit'])

                        mu, logstd, value, states = self.a2c_network(input_dict)
                        prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)

                    CommonAgent._actor_loss:
                        advantages = returns - values
                        -> a_loss = input_dict['advantages'] * torch.exp(input_dict['old_logp_actions'] - res_dict['prev_neglogp'])
                    CommonAgent._critic_loss:
                        -> c_loss = |input_dict['returns'] - res_dict['values']|
                    AMPAgent._disc_loss:
                        disc_loss_agent = bce([res_dict['disc_agent_logit'], res_dict['disc_agent_replay_logit']], 0)
                        disc_loss_demo = bce(res_dict['disc_demo_logit'], 1)
                        disc_grad_penalty = |torch.autograd.grad(res_dict['disc_demo_logit'], input_dict['amp_obs_demo'])|
                        -> disc_loss = disc_loss_agent + disc_loss_demo + disc_grad_penalty

                    
                    backward()+step()


    eval_info = self(IMAmpAgent).eval() ;
        repeat: // 
            obs_dict, r, done, info = self.env_eval_step(self.vec_env.env, action)
            done, info = self._post_step_eval(info, done.clone())
            until info['end']

    end



                    
// test
python phc/run_hydra.py learning=im_mcp exp_name=phc_shape_mcp_iccv epoch=-1 test=True env=env_im_getup_mcp \
robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl \ 
env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] env.num_envs=1  im_eval=True

// test pipeline
self.env = phc.env.tasks.vec_task_wrappers.VecTaskPythonWrapper
self.env.task = phc.env.tasks.humanoid_im_mcp_getup.HumanoidImMCPGetup(HumanoidImMCP(HumanoidIm(HumanoidAMPTask(HumanoidAMP((HumanoidAMP(Humanoid)))))))
learning.im_amp_players.IMAMPPlayerContinuous.run:
    for (n_games):
        for (self.max_steps):
            reset(last done)
            action = self.get_action(obs_dict, is_determenistic) // ([1, 4])
            obs_dict, r, done, info(['terminate', 'reward_raw', 'amp_obs', 'mpjpe', 'body_pos', 'body_pos_gt']) = self.env_step(self.env, action):
                self.env.task.step(action):
                    actions = torch.sum(action[:, :, None] * self.pnn(curr_obs), dim=1) // [1,4,1] x [1,4,69] = [1,69]
                    pre_physics_step(actions)
                    post_physics_step() -> info 
                        Humanoid.compute_humanoid_reset/compute_humanoid_im_reset:
                            terminated = torch.logical_and(fall_contact, fall_height)


            done = self._post_step(info, done.clone()):
                if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                    1- self.terminate_state => self.success_rate

                    if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):










// physics dynamics
/*
gym = gymapi.acquire_gym()
sim = gym.create_sim(sim_params)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())
env = gym.create_env(sim)
actor = gym.create_actor(env, asset)
gym.apply_rigid_body_force_tensors(sim)
gym.simulate(sim)
*/

// load data & model in env task
task, env = parse_task(args, cfg, cfg_train, sim_params)
task(HumanoidImMCPGetup) = eval(args.task)(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine, device_type=args.device, device_id=device_id, headless=args.headless)
HumanoidImMCPGetup.__mro__ = (
    <class 'phc.env.tasks.humanoid_im_mcp_getup.HumanoidImMCPGetup'>, 
    <class 'phc.env.tasks.humanoid_im_getup.HumanoidImGetup'>, 
    <class 'phc.env.tasks.humanoid_im_mcp.HumanoidImMCP'>, 
    <class 'phc.env.tasks.humanoid_im.HumanoidIm'>, 
    <class 'phc.env.tasks.humanoid_amp_task.HumanoidAMPTask'>, 
    <class 'phc.env.tasks.humanoid_amp.HumanoidAMP'>, 
    <class 'phc.env.tasks.humanoid.Humanoid'>, 
    <class 'phc.env.tasks.base_task.BaseTask'>, <class 'object'>)

HumanoidImGetup.__init__:
    self._generate_fall_states()

HumanoidImMCP.__init__: 
    self.pnn = load_pnn(pnn_ck, num_prim = self.num_prim, has_lateral = self.has_lateral, activation = self.z_activation, device = self.device)
    // num_prim: a2c_network.pnn.actors.x

HumanoidIm.__init__: []
    self.load_humanoid_configs(cfg) // smpl TODO
    

HumanoidAMP.__init__: 
    HumanoidAMP._load_motion(motion_file)
        self._motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)
        self(MotionLibBase).load_data(self.m_cfg.motion_file,  min_length = self.m_cfg.min_length, im_eval = self.m_cfg.im_eval)
        -> self._motion_data(['pose_quat_global'(1000, 24, 4), 'pose_quat', 'trans_orig', 'root_trans_offset', 'beta', 'gender', 'pose_aa'(1000, 72), 'fps']) // amass_isaac_standing_upright_slim

Humanoid.__init__: 
    self.load_humanoid_configs(cfg) // smpl
    self._setup_character_props(self.key_bodies)
    -> self._dof_obs_size, self._dof_size, self._num_self_obs
        HumanoidAMP._setup_character_props
        -> self._num_amp_obs_per_step

    self._setup_tensors()

BaseTask.__init__: 
    self.gym = gymapi.acquire_gym()
    self.create_sim():
        sim = self.gym.create_sim()
        Humanoid._create_ground_plane()
        Humanoid._create_envs(self.num_envs, self.cfg["env"]['env_spacing'], int(np.sqrt(self.num_envs))):
            // load humanoid
            -> self.humanoid_assets 
            for i in range(self.num_envs):
                // create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self._build_env(i, env_ptr, self.humanoid_assets[i])
    self.create_viewer()


self.env.task.step(action):
    if mcp:
        _, x_all = self.pnn(curr_obs)
        actions = torch.sum(action[:, :, None] * x_all, dim=1)
    pre_physics_step(actions)
        pd_tar = self._action_to_pd_targets(actions)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar)



AMPAgent.__mro__ = (
    <class 'phc_h1.learning.amp_agent.AMPAgent'>, 
    <class 'learning.common_agent.CommonAgent'>, 
    <class 'rl_games.algos_torch.a2c_continuous.A2CAgent'>, 
    <class 'rl_games.common.a2c_common.ContinuousA2CBase'>, 
    <class 'rl_games.common.a2c_common.A2CBase'>, 
    <class 'object'>)


A2CBase.__init__:
    self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
        RLGPUEnv.__init__: -> create_rlgpu_env:
        task, env = parse_task(args, cfg, cfg_train, sim_params)


    self.env_info = self.vec_env.get_env_info()





obs:

HumanoidIm._compute_observations:
    local: facig frame
    self_obs = Humanoid._compute_humanoid_obs(env_ids): // Humanoid.compute_humanoid_observations_smpl_max
        acquire_rigid_body_state_tensor
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        body_vel = self._rigid_body_vel[env_ids]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]
        if self_obs_v==2: // 
            hist
        if self_obs_v==3: // 
            acquire_force_sensor_tensor
        obs_list = [root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    task_obs = self._compute_task_obs(env_ids): // HumanoidIM.compute_imitation_observations_v6
        subset: self._track_bodies_id
        ro6D: quat_to_tan_norm [0,2]
        obs_list = [diff_local_body_pos, diff_local_body_rot, diff_local_body_vel, diff_local_body_ang_vel, local_ref_body_pos, local_ref_body_rot]
        

        

    obs = compute_humanoid_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._dof_obs_size, self._dof_offsets, body_shape_params, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs)
    obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)


reward:
HumanoidIM._compute_reward:
    HumanoidIM.compute_imitation_reward:
        k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
        w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

        k_pos, k_rot, k_vel, k_ang_vel = (100, 10, 0.1, 0.1)
        w_pos, w_rot, w_vel, w_ang_vel = (0.5, 0.3, 0.1, 0.1)

        r_body_pos = torch.exp(-k_pos * diff_body_pos)
        r_body_rot = torch.exp(-k_rot * diff_global_body_angle)
        r_vel = torch.exp(-k_vel * diff_global_vel)
        r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel)
        reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
        
        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
        power_reward = -self.power_coefficient * power # 0.0005
        power_reward[self.progress_buf <= 3] = 0