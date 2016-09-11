//
// Created by sauce on 11/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_CPP_ENV_H
#define SUPER_DUPER_OCTO_LAMP_CPP_ENV_H

enum DIRECTION { NORTH, SOUTH, EAST, WEST };

class MineCraftPseudoEnvironment {
 public:
  std::vector <std::vector<float>> &rewards;
  std::vector <std::vector<int>> &visibility_kernels;
  float psuedo_reward_multiplier;
  std::tuple<int, int, Direction> start_state;
  std::pair<int, int> end_state;

  MineCraftPseudoEnvironment(std::vector <std::vector<float>> &rewards,
                             std::vector <std::vector<int>> &visibility_kernels,
                             std::tuple<int, int, Direction> start_state,
                             std::pair<int, int> end_state,
                             float psuedo_rewards_multiplier)
      : rewards(rewards),
        visibility_kernels(visibility_kernels),
        start_state(start_state),
        end_state(end_state),
        psuedo_reward_multiplier(psuedo_rewards_multiplier) {

  }

};

#endif //SUPER_DUPER_OCTO_LAMP_CPP_ENV_H
