//
// Created by sauce on 12/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_ENV_H
#define SUPER_DUPER_OCTO_LAMP_ENV_H

#include <vector>
#include <tuple>

#define ENV_H_GAMMA 0.95f

class SingleMDP {
public:
  void SetEnv(const std::vector<std::vector<float>> &rewards, std::pair<int,int> start_state, std::pair<int,int> end_state);
  void Reset();
  std::tuple<std::pair<int,int>, float, bool> Step(int action);
  std::pair<int,int> GetEnvShape() {
    return env_shape;
  }
  int GetNumActions() {
    return actions.size();
  }
  std::pair<int,int> GetStartState() {
    return start_state;
  }
  float GetGamma() {
    return ENV_H_GAMMA;
  }

private:
  std::pair<int,int> cur_state;
  std::pair<int,int> end_state;
  std::pair<int,int> start_state;
  std::pair<int,int> env_shape;
  // Pointer to const reward matrix.
  std::vector<std::vector<float>> const *reward_matrix;
  const std::vector<std::pair<int,int>> actions = {std::make_pair(-1,0), std::make_pair(1,0), std::make_pair(0,1), std::make_pair(0,-1)};
};

void SingleMDP::SetEnv(const std::vector<std::vector<float>> &rewards, std::pair<int,int> start_state, std::pair<int,int> end_state) {
  reward_matrix = &rewards;
  start_state = start_state;
  end_state = end_state;
  cur_state = start_state;
  env_shape = std::make_pair(rewards.size(), rewards[0].size());
}

void SingleMDP::Reset() {
  this->cur_state = this->start_state;
}

std::tuple<std::pair<int,int>, float, bool> SingleMDP::Step(int action) {
  // Compute new state.
  int new_y = cur_state.first + actions[action].first;
  int new_x = cur_state.second + actions[action].second;
  new_y = (new_y < 0 ? 0 : (new_y > env_shape.first - 1 ? env_shape.first - 1 : new_y));
  new_x = (new_x < 0 ? 0 : (new_x > env_shape.second - 1 ? env_shape.second - 1 : new_x));
  this->cur_state = std::make_pair(new_y, new_x);

  // Compute reward signal.
  // Currently using only one component.
  // NOTE :: Can engineer and add a visibility matrix.
  float reward = (*this->reward_matrix)[new_y][new_x];

  return std::make_tuple(this->cur_state, reward, (this->cur_state == this->end_state));
}
#endif //SUPER_DUPER_OCTO_LAMP_ENV_H
