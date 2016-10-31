//
// Created by sauce on 11/9/16.
//

#include <vector>
#include <iostream>
#include <utility>
#include <random>
#include <tuple>

#ifndef SUPER_DUPER_OCTO_LAMP_CPP_MC_H
#define SUPER_DUPER_OCTO_LAMP_CPP_MC_H

enum Directions {SOUTH, EAST, NORTH, WEST};

// Function to generate primitive kernels
std::vector<std::vector<bool> > GetPrimitiveKernel(const int dir, const std::pair<int,int> kern_shape) {
  std::vector<std::vector<bool> > kernel(kern_shape.first, std::vector<bool>(kern_shape.second,false));
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

std::vector<std::vector<bool> > BitwiseAND(const std::vector<std::vector<bool> >& first,
                                          const std::vector<std::vector<bool> >& second) {
  std::vector<std::vector<bool> > ret = std::vector<std::vector<bool> >(first.size(),std::vector<bool>(first[0].size(),false));
  for(int i=0;i<first.size();i++)
    for(int j=0;j<first[0].size();j++)
      ret[i][j] = first[i][j] && second[i][j];

  return ret;
}
std::vector<std::vector<bool> > GetKernel(const int direction, const std::pair<int,int> kern_shape) {
  std::vector<std::vector<bool> > kernel;
  static bool first_run = true;
  static std::vector<std::vector<std::vector<bool> > > primitive_kernels = {GetPrimitiveKernel(0,kern_shape),
                                                                          GetPrimitiveKernel(1,kern_shape),
                                                                          GetPrimitiveKernel(2,kern_shape),
                                                                          GetPrimitiveKernel(3,kern_shape)};
  static std::vector<std::vector<std::vector<bool> > > final_kernels;
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
float GetDeltaAndUpdate(const std::vector<std::vector<bool> >& kernel, std::vector<std::vector<bool> > &visib, const std::pair<int,int> state, std::vector<std::vector<float> > pseudo ) {
  // Figure out i_min, i_max, j_min, j_max for kernel.
  int length = kernel.size();
  int num_pts = length/2;
  int y_min = std::max(state.first - num_pts, 0);
  int y_max = std::min(state.first + num_pts, (const int &) (visib.size() - 1));
  int x_min = std::max(state.second - num_pts, 0);
  int x_max = std::min(state.second + num_pts, (const int &) (visib[0].size() - 1));

  int num_changed = 0;
  float pseudo_reward = 0.0f;
  for(int y = y_min; y<= y_max; y++)
    for(int x = x_min; x<= x_max; x++) {
      num_changed += static_cast<int>(~visib[y][x] && kernel[(y-state.first)+num_pts][(x-state.second)+num_pts]);
      pseudo_reward += static_cast<int>(~visib[y][x] && kernel[(y-state.first)+num_pts][(x-state.second)+num_pts]) * pseudo[y][x];
      visib[y][x] = visib[y][x] || kernel[(y-state.first)+num_pts][(x-state.second)+num_pts];
    }

  return pseudo_reward;
}

// Python function call
std::vector<float> GetActionValueEstimate(const std::vector<std::vector<float> >& rewards,
                                          const std::vector<std::vector<float> >& pseudo,
                                          const std::tuple<int,int,int> start_state, const std::pair<int,int> end_state,
                                          const std::pair<int,int> kern_shape) {
  // Define global params here.
  std::pair<int,int> env_shape = std::make_pair(rewards.size(),rewards[0].size());
  const float gamma = 0.9;
  const auto& vis_kernels = {GetKernel(Directions::NORTH,kern_shape),GetKernel(Directions::SOUTH,kern_shape),
                             GetKernel(Directions::EAST,kern_shape),GetKernel(Directions::WEST,kern_shape)};

  /***************************** ENVIRONMENT ************************/
  // Define environment here.
  struct Environment {

    // Define env vars here.
    const std::vector<std::tuple<int,int,int> > actions = {std::make_tuple(0,1,0), std::make_tuple(1,0,0),std::make_tuple(0,-1,0),std::make_tuple(-1,0,0),
                          std::make_tuple(0,0,Directions::NORTH),std::make_tuple(0,0,Directions::SOUTH),std::make_tuple(0,0,Directions::EAST), std::make_tuple(0,0,Directions::WEST)};

    // Weight for pseudo rewards.
    const float pseudo_reward_multiplier = 0.0;
    // Env-specific rewards.
    std::vector<std::vector<float> > rewards;
    // Env-specific pseudo-rewards.
    std::vector<std::vector<float> > pseudo;
    // Portion of the map that is seen by the agent.
    std::vector<std::vector<bool> > seen;
    // All 4 possible visibility kernels.
    std::vector< std::vector< std::vector<bool> > > vis_kernels;
    // Kernel Shape(H x W).
    std::pair<int,int> kern_shape;
    // Environment Shape(H x W).
    std::pair<int,int> env_shape;
    // Current state.
    std::tuple<int,int,int> state;
    // End state.
    std::pair<int,int> end_state;
    // Current direction.
    Directions cur_direction;

    // Constructor to initialize env.
    Environment(const std::vector<std::vector<float> >& rewards,
                const std::vector<std::vector<float> >& pseudo,
                const std::tuple<int,int,int> start_state, const std::pair<int,int> end_state,
                const std::pair<int,int> kern_shape) {
      this->rewards = rewards;
      this->pseudo = pseudo;
      env_shape = std::make_pair(rewards.size(), rewards[0].size());
      seen = std::vector<std::vector<bool> >(env_shape.first,std::vector<bool>(env_shape.second,false));
      this->vis_kernels = {GetKernel(Directions::SOUTH,kern_shape),GetKernel(Directions::EAST,kern_shape),
                     GetKernel(Directions::NORTH,kern_shape),GetKernel(Directions::WEST,kern_shape)};
      this->kern_shape = kern_shape;
      state = start_state;
      this->end_state = end_state;
      cur_direction = static_cast<Directions>( std::get<2>(start_state) );
    }

    // Play() for one-step.
    std::tuple<std::tuple<int,int,int>,float,bool> play(int action) {
      // Obtain action taken.
      const auto action_taken = actions[action];

      int first, second, third;
      std::tie( first, second, third ) = action_taken;
      // Update coordinates/state.
      int s_first = std::get<0>(state);
      int s_second = std::get<1>(state);

      int y_coord = (s_first+first < 0 ? 0 : ( (s_first+first >= env_shape.first) ? (env_shape.first-1) : (s_first+first) ) );
      int x_coord = (s_second+second < 0 ? 0 : ( (s_second+second >= env_shape.second) ? (env_shape.second-1) : (s_second+second) ) );

      // Update the seeing direction.
      cur_direction = static_cast<Directions>(action < 4 ? cur_direction : action % 4);
      const auto new_state = std::make_tuple(y_coord, x_coord, static_cast<int>(cur_direction));


      // Calculate change in visibility.
      float delta_vis = GetDeltaAndUpdate(this->vis_kernels[(action < 4 ? cur_direction : action % 4)], seen, std::make_pair( std::get<0>(new_state),std::get<1>(new_state) ), pseudo);

      // Produce a pseudo-reward signal for change in visibility.
      float pseudo_reward = delta_vis;

      // Compute total reward signal (0 actual reward if in the same state)
      float total_reward = (action < 4 ? (new_state == state ? -50 : rewards[y_coord][x_coord]) : 0) + pseudo_reward;

      // Update state.
      state = new_state;
      int t0 = std::get<0>(state);
      int t1 = std::get<1>(state);
      // Return (new state, reward, is_completed)
      return std::make_tuple(new_state, total_reward-1, ( std::make_pair( t0, t1) == end_state) );
    };
  };

  /***************************** AGENT ************************/
  // Define agent params here.
  const int num_trajectories = 200;
  const float alpha = 0.1;
  const int max_num_steps = 200;
  const int num_actions = 8;
  const int num_directions = 4;

  std::vector< std::vector<std::vector<std::vector<float> > > > q_values = std::vector<std::vector<std::vector<std::vector<float> > > >(env_shape.first,std::vector<std::vector<std::vector<float> > >(env_shape.second,std::vector<std::vector<float>>( num_directions, std::vector<float>( num_actions,0.0f))) );;

  // Define agent behaviour here.
  std::random_device                  rand_dev;
  std::mt19937                        generator(rand_dev());
  std::uniform_int_distribution<int>  distr(0, q_values[0][0][0].size() - 1);

  std::uniform_int_distribution<int>  distr2(0, env_shape.first-1);
  std::uniform_int_distribution<int>  distr3(0, env_shape.second-1);
  std::uniform_int_distribution<int>  distr4(0, num_directions-1);
  Environment env(rewards,pseudo,start_state,end_state,kern_shape);

  // Obtain trajectories
  std::vector<std::vector<std::tuple<std::tuple<int,int,int>, int, float, std::tuple<int,int,int> > > > trajectories;
#pragma omp declare reduction (merge : std::vector<int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge: trajectories)
  for(int i=0;i<num_trajectories;i++) {
    // Start state (2-tuple) [int]
    std::vector < std::tuple < std::tuple < int, int, int >, int, float, std::tuple < int, int, int> > > cur_trajectory;

    std::tuple<int,int,int> start_state = std::make_tuple(distr3(generator),distr2(generator),distr4(generator));

    Environment cur_env(rewards, pseudo, start_state, end_state, kern_shape);
    std::tuple<int, int, int> cur_state(start_state);
    bool is_completed = false;
    int num_steps = 0;
    while (!is_completed && num_steps < max_num_steps) {
      // Choose a random action.
      int cur_action = distr(generator);

      // Play the action.
      const auto &env_response = env.play(cur_action);

      std::tuple<int, int, int> new_state = std::get<0>(env_response);
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

    for( auto iter = traj.rbegin(); iter != traj.rend(); iter++) {

      // Obtain (s,a,r,s) values
      std::tuple<int,int,int> new_state,old_state;
      float reward;
      int action;
      std::tie(old_state, action, reward, new_state) = *iter;

      //printf("%d,%d %d %f %d,%d\n", old_state.first, old_state.second, action, reward, new_state.first, new_state.second);
      // Obtain target
      float target = reward + gamma * (*std::max_element(q_values[std::get<0>(new_state)][std::get<1>(new_state)][std::get<2>(new_state)].begin(), q_values[std::get<0>(new_state)][std::get<1>(new_state)][std::get<2>(new_state)].end()));

      // Obtain TD error
      float TD_error = (target - q_values[std::get<0>(old_state)][std::get<1>(old_state)][std::get<2>(old_state)][action]);

      // Perform Q update
#pragma omp atomic
      q_values[std::get<0>(old_state)][std::get<1>(old_state)][std::get<2>(old_state)][action] += alpha * TD_error;
    }
  }
  for( int action = 0; action < 8; action++ ) {
    for (auto row : q_values) {
      for (auto cell : row) {
        //for( auto action : cell )
        float f = 0;
        for( int k = 0; k < 4; k++ ){
          f += cell[k][action];
        }
        f /= 8;
        printf("%5.3f\t", f);
      }
      printf("\n");
    }
    printf("\n");
  }

  std::vector<std::string> movement_symbols = {"->","v","<-","^"};
  /*for (auto row : q_values) {
    for (auto cell : row) {
        //for( auto action : cell )
      int index = (int) (std::max_element( cell.begin(), cell.begin() + 4 ) - cell.begin());
      printf( "%s\t", movement_symbols[ index ].c_str() );


    }
    printf("\n");
  }*/
  printf("\n");


  /*for (auto row : q_values) {
    for (auto cell : row) {
      //for( auto action : cell
      printf( "%6.3f\t", *std::max_element(cell.begin(), cell.begin() + 4 ) );


    }
    printf("\n");
  }*/
  printf("\n");


  return q_values[std::get<0>(start_state)][std::get<1>(start_state)][std::get<2>(start_state)];
}

#endif //SUPER_DUPER_OCTO_LAMP_CPP_MC_H
