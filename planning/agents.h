//
// Created by sauce on 12/9/16.
//

#ifndef SUPER_DUPER_OCTO_LAMP_AGENTS_H
#define SUPER_DUPER_OCTO_LAMP_AGENTS_H

#include <algorithm>
#include <numeric>
#include <random>
#include "environment/env.h"

#define Q_AGENT_ALPHA 0.05f

namespace {
  // Softmax Selection
  int SoftmaxSelection(std::vector<float> values) {
    // Compute exp of values
    for(auto& val : values)
      val = std::exp(val);

    // Compute sum of values
    sum_values = std::accumulate(values.begin(), values.end(), 0.0f);

    // Normalize the vector
    for(auto& val : values)
      val /= sum_values;

    // Find cumsum of vector
    float cumsum = 0.0f;
    for(auto& val : values) {
      cumsum += val;
      val = cumsum;
    }

    // Init once
    {
      static std::random_device rand_dev;
      static std::mt19937 generator(rand_dev());
      static std::uniform_int_distribution<float> distr(0.0f, 1.0f);
    }

    // Sample a random value from uniform distribution
    float chooser = distr(generator);

    // Find matching index.
    int idx = 0;
    for(const auto val : values) {
      if(val >= chooser)
        return idx;

      idx++;
    }

    // If everything fails, return last action.
    return (values.size() - 1);
  }

  // Greedy Selection
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
  virtual ActionType Step(const StateType& prev_state, const float reward) = 0;
};

class QAgent : public Agent<std::pair<int,int>,std::pair<int,bool>> {
public:
  // Init function.
  void Init(const SingleMDP& env);

  // Step function; Returns num_steps taken if prev_state = <-1,-1>
  int Step(const std::pair<int,int>& new_state, const float reward);

  // Reset agent to start state, preserve value functions.
  void SoftReset();

  // Reset agent to start state, discard value functions.
  void HardReset();

private:
  int num_steps;
  float gamma;
  std::vector<std::vector<std::vector<float>>> q_values;
  std::pair<int,int> prev_state;
  int prev_action;
};

void QAgent::Init(const SingleMDP& env) {
  const auto env_shape = env.GetEnvShape();
  const int num_actions = env.GetNumActions();
  this->q_values = std::vector<std::vector<std::vector<float>>>(env_shape.first,std::vector<std::vector<float>>(env_shape.second,std::vector<float>(num_actions,0.0f)));
  this->gamma = env.GetGamma();

  this->prev_state = std::make_pair(-1,-1);
  this->prev_action = -1;
}

int QAgent::Step(const std::pair<int,int>& new_state, const float reward) {
  if(new_state.first == new_state.second == -1){
    return this->num_steps;
  }
  // If not starting action
  if(prev_action != -1) {
    // Perform Q update first.
    const auto& req_q_values = q_values[new_state.first][new_state.second];
    float target = reward + this->gamma * (*std::max_element(req_q_values.begin(),req_q_values.end()));
    float Td_error = (target - q_values[prev_state.first][prev_state.second][prev_action]);
    q_values[prev_state.first][prev_state.second][prev_action] += Q_AGENT_ALPHA*Td_error;
  }

  // Softmax selection over Q-values
  prev_state = new_state;
  prev_action = SoftmaxSelection(q_values[new_state.first][new_state.second]);
  return prev_action;
}

void QAgent::SoftReset() {
  this->prev_state = make_pair(-1,-1);
  this->prev_action = -1;
}

void QAgent::HardReset() {
  for(auto& row : q_values)
    for(auto& col : row)
      for(auto& action : col)
        action = 0.0f;
  this->prev_state = make_pair(-1,-1);
  this->prev_action = -1;
}
#endif //SUPER_DUPER_OCTO_LAMP_AGENTS_H
