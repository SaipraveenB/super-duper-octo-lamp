//
// Created by sauce on 11/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_CPP_MC_H
#define SUPER_DUPER_OCTO_LAMP_CPP_MC_H

class MCPlanner {
    MCPlanner(MinecraftPsuedoEnvironment env, const int num_trajs, const float learning_rate, const float gamma,
              const int max_num_steps) : env(env), num_trajectories(num_trajs), alpha(learning_rate), gamma(gamma),
                                         max_num_steps(max_num_steps) {
        const auto &env_shape = env.GetShape();
        const int num_actions = env.GetNumActions();
        q_values = std::vector<std::vector<std::vector<float>>>(env_shape[0],std::vector<std::vector<float>>(env_shape[1],std::vector<float>(num_actions,0.0f)));
    }

    const std::vector<float> plan()
    {
        std::random_device                  rand_dev;
        std::mt19937                        generator(rand_dev());
        std::uniform_int_distribution<int>  distr(0, q_values[0][0].size() - 1);

        // Obtain trajectories
        std::vector<std::vector<std::tuple<std::pair<int,int>, int, float, std::pair<int,int>>>> trajectories;
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge: trajectories)
        for(int i=0;i<num_trajectories;i++) {
            // Start state (2-tuple) [int]
            std::vector < std::tuple < std::pair < int, int >, int, float, std::pair < int, int >> > cur_trajectory;
            MinecraftPseudoEnvironment cur_env = env;
            std::pair<int, int> cur_state = env.GetStartState();
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
        // Construct Q-values
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

        return q_values[start_state.first][start_state.second];
    }


private:
    MinecraftPsuedoEnvironment env;
    const int num_trajectories;
    const float alpha;
    const float gamma;
    const int max_num_steps;

    std::vector <std::vector<std::vector < int>>>
    q_values;
};

#endif //SUPER_DUPER_OCTO_LAMP_CPP_MC_H
