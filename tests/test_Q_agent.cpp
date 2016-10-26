#include "planning/agents.h"
#include "environment/env.h"

#include <utility>
#include <vector>

class QAgentExperiment {
public:
  QAgentExperiment(const std::vector<std::vector<float>> &rewards, std::pair<int, int> start_state, std::pair<int, int> end_state);
  std::vector<std::pair<float, float>> RunMultipleExperiments(const int num_reps, const int max_num_iterations, const int max_num_steps);

private:
  std::vector<std::pair<int, float>> RunSingleExperiment(const int max_num_iterations, const int max_num_steps, QAgent& agent, SingleMDP& env);
  QAgent agent;
  SingleMDP env;
};

QAgentExperiment::QAgentExperiment(const std::vector<std::vector<float>> &rewards, std::pair<int, int> start_state, std::pair<int, int> end_state) {
  this->env = SingleMDP();
  this->env.SetEnv(rewards, start_state, end_state);
  this->agent = QAgent();
  this->agent.Init(this->env);
}

std::vector<std::pair<int, float>> QAgentExperiment::RunSingleExperiment(const int max_num_iterations, const int max_num_steps, QAgent& agent, SingleMDP& env) {
  std::vector<std::pair<int, float>> iter_avg_reward_training(max_num_iterations, std::make_pair(0,0.0f));

  for(int i=0;i<max_num_iterations;i++) {
    int num_steps = 0;
    float rewards = 0.0f;
    // First step
    int action = agent.Step(env.GetStartState(),0.0f);

    while(num_steps < max_num_steps) {
      // Get new state from action.
      auto new_state = env.Step(action);

      // Update tracking params.
      rewards += std::get<1>(new_state);
      num_steps++;

      // If reached completion, break.
      if(std::get<2>(new_state))
        break;

      action = agent.Step(std::get<0>(new_state));
    }

    // Reset env, soft reset agent.
    env.Reset();
    agent.SoftReset();

    // Update array.
    iter_avg_reward_training[i] = make_pair(num_steps, rewards/num_steps);
  }

  return iter_avg_rewards;
}

std::vector<std::pair<float, float>> QAgentExperiment::RunMultipleExperiments(const int num_reps, const int max_num_iterations, const int max_num_steps) {
  std::vector<std::pair<float, float>> avg_steps_rewards(max_num_iterations, std::make_pair(0.0f,0.0f));
#pragma omp parallel for
  for(int i=0;i<num_reps;i++) {
    QAgent inside_agent(agent);
    SingleMDP inside_env(env);
    const auto steps_rewards = this->RunSingleExperiment(max_num_iterations, max_num_steps, inside_agent, inside_env);

    for(int j=0;j<max_num_iterations;j++)
#pragma omp critical
        {
          avg_steps_rewards[j].first += steps_rewards[j].first;
          avg_steps_rewards[j].second += steps_rewards[j].second;
        }
  }
  for(auto& vals : avg_steps_rewards) {
    vals[i].first /= num_reps;
    vals[i].second /= num_reps;
  }

  return avg_steps_rewards;
}

//TODO(SaiPraveenB) : Make Python call to this function
std::vector<std::pair<float,float>> GetData(const std::vector<std::vector<float>> &rewards, std::pair<int,int> start_state, std::pair<int,int> end_state, const int num_reps, const int max_num_iterations, const int max_num_steps) {
  QAgentExperiment q_agent_exp(rewards, start_state, end_state);

  auto stats = q_agent_exp.RunMultipleExperiments(num_reps, max_num_iterations, max_num_steps);

  return stats;
}
