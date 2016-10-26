#include "planning/agents.h"
#include "environment/env.h"

#include <vector>

class QAgentExperiment {
public:
  QAgentExperiment(const std::vector<std::vector<float>> &rewards, std::pair<int,int> start_state, std::pair<int,int> end_state);
  std::vector<std::pair<float,float>> RunMultipleExperiments(const int num_repetitions, const int max_num_iterations, const int max_num_steps);

private:
  std::vector<std::pair<int,float>> RunSingleExperiment(const int max_num_iterations, const int max_num_steps, QAgent& agent, SingleMDP& env);
  QAgent agent;
  SingleMDP env;
}

QAgentExperiment::QAgentExperiment() {
  this->env = SingleMDP();
  this->env.SetEnv(rewards, start_state, end_state);
  this->agent = QAgent();
  this->agent.Init(this->env);
}

std::vector<std::pair<int,float>> QAgentExperiment::RunSingleExperiment(const int max_num_iterations, const int max_num_steps, QAgent& agent, SingleMDP& env) {
  std::vector<std::pair<int,float>> iter_avg_rewards(max_num_steps,std::make_pair(0,0.0f));

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

      // Send values to agent for next action, CONDITIONALLY.
      if(!std::get<2>(new_state) ) {
        action = agent.Step(std::get<0>(new_state));
      }
      else
        break;
    }
    // Reset env, soft reset agent.
    agent.SoftReset();
    env.Reset();

    // Update array.
    iter_avg_rewards[i] = make_pair(num_steps, rewards/num_steps);
  }

  return iter_avg_rewards;
}

std::vector<std::pair<float,float>> QAgentExperiment::RunMultipleExperiments(const int num_repetitions, const int max_num_iterations, const int max_num_steps) {
  std::vector<std::pair<float,float>> avg_steps_rewards(max_num_iterations, std::make_pair(0.0f,0.0f));
#pragma omp parallel for
  for(int i=0;i<num_repetitions;i++) {
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
    vals[i].first /= num_repetitions;
    vals[i].second /= num_repetitions;
  }

  return avg_steps_rewards;
}

//TODO(SaiPraveenB) : Make Python call to this function
std::vector<std::pair<float,float>> GetData(const std::vector<std::vector<float>> &rewards, std::pair<int,int> start_state, std::pair<int,int> end_state, const int num_repetitions, const int max_num_iterations, const int max_num_steps) {
  QAgentExperiment q_agent_exp(rewards, start_state, end_state);

  return q_agent_exp.RunMultipleExperiments(num_repetitions, max_num_iterations, max_num_steps);
}
