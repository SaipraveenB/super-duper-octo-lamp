//
// Created by sauce on 12/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_AGENTS_H
#define SUPER_DUPER_OCTO_LAMP_AGENTS_H

#include "environment/env.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#define Q_AGENT_ALPHA 0.05f

namespace {
  // Softmax Selection
  int SoftmaxSelection(std::vector<float> values) {
    // Compute exp of values
    for(auto& val : values)
      val = std::exp(val);

    // Compute sum of values
    sum_values = std::accumulate(values.begin(), values.end(), 0.0f);

    // Normalize the vector; Find cumulative sum of probs.
    float cumsum = 0.0f;
    for(auto& val : values) {
      cumsum += (val/sum_values);
      val = cumsum;
    }

    // Init RNGs once
    {
      static std::random_device rand_dev;
      static std::mt19937 generator(rand_dev());
      static std::uniform_int_distribution<float> distr(0.0f, 1.0f);
    }

    // Sample a random value from uniform distribution
    float chooser = distr(generator);

    // Find matching index.
    int idx = 0;
    for(const float val : values) {
      if(val >= chooser)
        return idx;

      idx++;
    }

    // If everything fails (for some random reason), return 'last' action by default.
    return (idx - 1);
  }

  // Greedy Selection (if needed)
  int GreedySelection(std::vector<float> values) {
    return (std::max_element(values.begin(),values.end()) - values.begin());
  }
} // namespace

template<typename StateType, typename ActionType>
class Agent {
public:
  // Init function.
  virtual void Init(const Env& env) = 0;

  // Step function.
  virtual ActionType Step(const StateType& new_state, const float reward) = 0;
};

// Q Agent for state type = std::pair<int, int> and action type = int
class QAgent : public Agent<std::pair<int, int>, int> {
public:
  // Init function.
  void Init(const SingleMDP& env);

  // Step function.
  // NOTE :: If prev_state=<-1,-1> or prev_action = -1, doesn't perform Q-update.
  int Step(const std::pair<int, int>& new_state, const float reward);

  // Reset agent to start state, preserve value functions.
  void SoftReset();

  // Reset agent to start state, discard value functions.
  void HardReset();

private:
  int num_steps;
  float gamma;
  std::vector<std::vector<std::vector<float>>> q_values;
  std::pair<int, int> prev_state;
  int prev_action;
};

void QAgent::Init(const SingleMDP& env) {
  const auto env_shape = env.GetEnvShape();
  const int num_actions = env.GetNumActions();
  this->q_values = std::vector<std::vector<std::vector<float>>>(env_shape.first, std::vector<std::vector<float>>(env_shape.second, std::vector<float>(num_actions, 0.0f)));
  this->gamma = env.GetGamma();

  this->prev_state = std::make_pair(-1,-1);
  this->prev_action = -1;
}

int QAgent::Step(const std::pair<int, int>& new_state, const float reward) {
  if(new_state.first == new_state.second == -1){
    return this->num_steps;
  }
  // When action != first action.
  if(prev_action != -1) {
    // Perform Q update.
    const auto& req_q_values = q_values[new_state.first][new_state.second];
    const float target = reward + this->gamma * (*std::max_element(req_q_values.begin(), req_q_values.end()));
    const float td_error = (target - q_values[prev_state.first][prev_state.second][prev_action]);
    q_values[prev_state.first][prev_state.second][prev_action] += Q_AGENT_ALPHA * td_error;
  }

  // Softmax selection using current Q-values.
  prev_state = new_state;
  prev_action = SoftmaxSelection(q_values[new_state.first][new_state.second]);
  return prev_action;
}

void QAgent::SoftReset() {
  // Clear state, action.
  this->prev_state = make_pair(-1,-1);
  this->prev_action = -1;
}

void QAgent::HardReset() {
  // Clear Q-values.
  for(auto& row : q_values)
    for(auto& col : row)
      for(auto& action : col)
        action = 0.0f;

  // Clear state, action.
  this->prev_state = make_pair(-1,-1);
  this->prev_action = -1;
}

#endif //SUPER_DUPER_OCTO_LAMP_AGENTS_H
