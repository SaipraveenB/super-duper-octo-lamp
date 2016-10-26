//
// Created by sauce on 11/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_CPP_MC_H
#define SUPER_DUPER_OCTO_LAMP_CPP_MC_H

// Function to generate primitive kernels
std::vector<std::vector<bool>> GetPrimitiveKernel(const int dir, const std::pair<int,int> kern_shape) {
    std::vector<std::vector<bool>> kernel(kern_shape.first, std::vector<bool>(false,kern_shape.second));
    switch(dir) {
        case 0 :
            for(int i=0;i<kern_shape.first;i++)
                for(int j=0;j<=i;j++)
                    kernel[i][j] = true;
            break;
        case 1 :
            for(int i=0;i<kern_shape.first;i++)
                for(int j=0;j<=kern_shape.first-1 - i;j++)
                    kernel[i][j] = true;
            break;
        case 2 :
            for(int i=0;i<kern_shape.first;i++)
                for(int j=i;j<=kern_shape.first - 1;j++)
                    kernel[i][j] = true;
            break;
        case 3 :
            for(int i=0;i<kern_shape.first;i++)
                for(int j=kern_shape.first-1 - i;j<=kern_shape.first - 1;j++)
                    kernel[i][j] = true;
            break;
    }

    return kernel;
}

std::vector<std::vector<bool>> BitwiseAND(const std::vector<std::vector<bool>>& first,
                                          const std::vector<std::vector<bool>>& second) {
    std::vector<std::vector<bool>> ret = std::vector<std::vector<bool>>(first.size(),std::vector<bool>(first[0].size(),false));
    for(int i=0;i<first.size();i++)
        for(int j=0;j<first[0].size();j++)
            ret[i][j] = first[i][j] && second[i][j];

    return ret;
}
std::vector<std::vector<bool>> GetKernel(const int direction, const std::pair<int,int> kern_shape) {
    std::vector<std::vector<bool>> kernel;
    static bool first_run = true;
    static std::vector<std::vector<std::vector<bool>>> primitive_kernels = {GetPrimitiveKernel(0,kern_shape),
                                                                            GetPrimitiveKernel(1,kern_shape),
                                                                            GetPrimitiveKernel(2,kern_shape),
                                                                            GetPrimitiveKernel(3,kern_shape)};
    static std::vector<std::vector<std::vector<bool>>> final_kernels;
    if(first_run) {
        final_kernels = {BitwiseAND(primitive_kernels[1], primitive_kernels[2]),
                         BitwiseAND(primitive_kernels[0], primitive_kernels[3]),
                         BitwiseAND(primitive_kernels[2], primitive_kernels[3]),
                         BitwiseAND(primitive_kernels[0], primitive_kernels[1])};
        first_run = false;
    }

    return final_kernels[direction];
}

// Make sure kernel.size() == kernel[0].size() == EVEN
int GetDeltaAndUpdate(const std::vector<std::vector<bool>>& kernel, std::vector<std::vector<bool>> &visib, const std::pair<int,int> state) {
    // Figure out i_min, i_max, j_min, j_max for kernel.
    int length = kernel.size();
    int num_pts = length/2;
    int y_min = std::max(state.first - num_pts, 0);
    int y_max = std::min(state.first + num_pts, visib.size() - 1);
    int x_min = std::max(state.second - num_pts, 0);
    int x_max = std::min(state.second + num_pts, visib[0].size() - 1);

    int num_changed = 0;
    for(int y = y_min; y<= y_max; y++)
        for(int x = x_min; x<= x_max; x++) {
            num_changed += static_cast<int>(~visib[state.first + y][state.second + x] && kernel[y][x]);
            visib[state.first + y][state.second + x] |= kernel[y][x];
        }

    return num_changed;
}

// Python function call
std::vector<float> GetActionValueEstimate(const std::vector<std::vector<float>>& rewards,
                                        const std::pair<int,int> start_state, const std::pair<int,int> end_state,
                                        const std::pair<int,int> kern_shape) {
    // Define global params here.
    std::pair<int,int> env_shape = std::make_pair(rewards.size(),rewards[0].size());
    const float gamma = 0.9;
    const auto& vis_kernels = {GetKernel(NORTH,kern_shape),GetKernel(SOUTH,kern_shape),
                               GetKernel(EAST,kern_shape),GetKernel(WEST,kern_shape)};

    /***************************** ENVIRONMENT ************************/
    // Define environment here.
    struct Environment {
        const enum Directions {NORTH, SOUTH, EAST, WEST};

        // Define env vars here.
        const auto actions = { make_tuple(0,1,0), make_tuple(1,0,0),make_tuple(0,-1,0),make_tuple(-1,0,0),
                               make_tuple(0,0,NORTH),make_tuple(0,0,SOUTH),make_tuple(0,0,EAST), make_tuple(0,0,WEST)};

        // Weight for pseudo rewards.
        const float pseudo_reward_multiplier = 1.0;
        // Env-specific rewards.
        const std::vector<std::vector<float> >& rewards;
        // Portion of the map that is seen by the agent.
        std::vector<std::vector<bool> > seen;
        // All 4 possible visibility kernels.
        std::vector< std::vector< std::vector<bool> > > vis_kernels;
        // Kernel Shape(H x W).
        std::pair<int,int> kern_shape;
        // Environment Shape(H x W).
        std::pair<int,int> env_shape;
        // Current state.
        std::pair<int,int> state;
        // End state.
        const std::pair<int,int> end_state;
        // Current direction.
        Directions cur_direction;

        // Constructor to initialize env.
        Environment(const std::vector<std::vector<float>>& rewards,
                    const std::pair<int,int> start_state, const std::pair<int,int> end_state,
                    const std::pair<int,int> kern_shape) {
            rewards = rewards;
            env_shape = std::make_pair(rewards.size(), rewards[0].size());
            seen = std::vector<std::vector<bool> >(env_shape.first,std::vector<bool>(env_shape.second,false));
            vis_kernels = {GetKernel(NORTH,kern_shape),GetKernel(SOUTH,kern_shape),
                           GetKernel(EAST,kern_shape),GetKernel(WEST,kern_shape)};
            kern_shape = kern_shape;
            state = start_state;
            end_state = end_state;
            cur_direction = NORTH;
        }

        // Play() for one-step.
        std::tuple<std::pair<int,int>,float,bool> play(int action) {
            // Obtain action taken.
            const auto action_taken = actions[action];

            // Update coordinates/state.
            int y_coord = (state.first+action_taken.first < 0 ? 0 : state.first+action_taken.first >= env_shape.first ? env_shape.first-1 : state.first+action_taken.first);
            int x_coord = (state.second+action_taken.second < 0 ? 0 : state.second+action_taken.second >= env_shape.second? env_shape.second-1 : state.first+action_taken.second);
            const auto new_state = std::make_pair(y_coord, x_coord);

            // Update the seeing direction.
            cur_direction = (action < 4 ? cur_direction : action % 4);

            // Calculate change in visibility.
            int delta_vis = GetDeltaAndUpdate(vis_kernels[(action < 4 ? cur_direction : action % 4)], seen, new_state);

            // Produce a pseudo-reward signal for change in visibility.
            float pseudo_reward = delta_vis*pseudo_reward_multiplier;

            // Compute total reward signal (0 actual reward if in the same state)
            float total_reward = (action < 4 ? (new_state == state ? 0 : rewards[y_coord][x_coord]) : 0) + pseudo_reward;

            // Update state.
            state = new_state;

            // Return (new state, reward, is_completed)
            return std::make_tuple(new_state, total_reward, (state == end_state));
        };
    };

    /***************************** AGENT ************************/
    // Define agent params here.
    const int num_trajectories = 200;
    const float alpha = 0.05;
    const int max_num_steps = 400;
    const int num_actions = 8;

    std::vector<std::vector<std::vector<float>>> q_values = q_values = std::vector<std::vector<std::vector<float>>>(env_shape[0],std::vector<std::vector<float>>(env_shape[1],std::vector<float>(num_actions,0.0f)));;

    // Define agent behaviour here.
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(0, q_values[0][0].size() - 1);
    const Environment env(rewards,start_state,end_state,kern_shape);

    // Obtain trajectories
    std::vector<std::vector<std::tuple<std::pair<int,int>, int, float, std::pair<int,int>>>> trajectories;
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge: trajectories)
    for(int i=0;i<num_trajectories;i++) {
        // Start state (2-tuple) [int]
        std::vector < std::tuple < std::pair < int, int >, int, float, std::pair < int, int >> > cur_trajectory;
        Environment cur_env(env);
        std::pair<int, int> cur_state(start_state);
        bool is_completed = false;
        int num_steps = 0;
        while (!is_completed && num_steps < this->max_num_steps) {
            // Choose a random action.
            int cur_action = distr(generator);

            // Play the action.
            const auto &env_response = env.Play(cur_action);

            std::pair<int, int> new_state = std::get<0>(env_response);
            float reward = std::get<1>(env_response);
            is_completed = std::get<2>(env_response);

            // Add (s,a,r,s) to experience
            cur_trajectory.push_back(std::make_tuple(cur_state, cur_action, reward, new_state));

            // Update params
            num_steps++;
            cur_state = new_state;
        }
        trajectories.push_back(cur_trajectory);
    }
    // Compute Q-values
#pragma omp parallel for
    for(const auto& traj : trajectories) {
        for(const auto iter = traj.rbegin(); iter != traj.rend(); iter++) {
            // Obtain (s,a,r,s) values
            std::pair<int,int> new_state,old_state;
            float reward;
            int action;
            std::tie(old_state, action, reward, new_state) = *iter;

            // Obtain target
            float target = reward + gamma * (*std::max_element(q_values[old_state.first][old_state.second].begin(), q_values[old_state.first][old_state.second].end()));

            // Obtain TD error
            float TD_error = (target - q_values[old_state.first][old_state.second][action]);

            // Perform Q update
#pragma omp atomic
            q_values[old_state.first][old_state.second] += alpha * TD_error;
        }
    }
    for( int action = 0; action < 8; action ++ ) {
      for (auto row : q_values) {
        for (auto cell : row) {
          printf("%5.3f\t", cell);
        }
        printf("\n");
      }
    }
    return q_values[start_state.first][start_state.second];
}

#endif //SUPER_DUPER_OCTO_LAMP_CPP_MC_H
