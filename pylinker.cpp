
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>

#include <vector>
#include <utility>


#include "planner.h"
#include "planner_vi.h"
#include "planner_vip.h"

/*
char const* greet()
{
   return "hello, world";
}
char const* meet()
{
    return "yay";
}

int get_first( boost::python::list& lst ){
    return len( lst );
}

BOOST_PYTHON_MODULE(hello_ext)
{
    using namespace boost::python;
    def("greet", greet);
    def("meet", meet);
    def("get_first", get_first);
}
*/
using namespace std;
using namespace boost::python;

template<typename T>
vector< vector<T> > convert(numeric::array a){

    vector< vector<T> > out;
    int a_len = len(a);
    for( int i = 0; i < a_len; i ++ ){
        numeric::array k = extract<numeric::array>(a[i]);
        int k_len = len(k);
        vector<T> sub;
        for( int j = 0; j < k_len; j++ ) {
          T t = static_cast<T>( extract<double>(k[j]) );
          sub.push_back(t);
        }

        out.push_back(sub);
    }

    return out;

}

// Functions to demonstrate extraction
/*int get_best_action(numeric::array rewards, numeric::array pseudo, boost::python::tuple kernel_size, boost::python::tuple start_state) {

    // Access a built-in type (an array)
    //boost::python::numeric::array a = data;
    // Need to <extract> array elements because their type is unknown.
    //a = extract<boost::python::numeric::array>(a[0]);
    //return extract<int> ( a[2] );
    vector<vector<float>> f_rewards = convert<float>( rewards );
    //vector<vector<bool>> f_mask = convert<bool>( mask );

    vector<vector<float>> f_pseudo = convert<float>( pseudo );
    //vector<vector<bool>> f_pmask = convert<bool>( pmask );
    
    int ks0 = boost::python::extract<int>(kernel_size[0]);
    int ks1 = boost::python::extract<int>(kernel_size[1]);
    
    int ss0 = boost::python::extract<int>(start_state[0]);
    int ss1 = boost::python::extract<int>(start_state[1]);
  int ss2 = boost::python::extract<int>(start_state[2]);

    std::pair<int,int> p_kern = std::make_pair( ks0, ks1 );
    std::tuple<int,int,int> p_ss = std::make_tuple( ss0, ss1, ss2 );

  std::cout<<f_rewards.size()<<std::endl;
  std::flush(std::cout);
  std::cout<<f_rewards[0].size()<<std::endl;
  std::flush(std::cout);
  std::cout<<f_pseudo.size()<<std::endl;
  std::flush(std::cout);
  std::cout<<f_pseudo[0].size()<<std::endl;
  std::flush(std::cout);


  std::cout<<ks0<<std::endl;
  std::cout<<ks1<<std::endl;
  std::cout<<ss0<<std::endl;
  std::cout<<ss1<<std::endl;
  std::flush(std::cout);

  for( auto row : f_pseudo ){
    for( auto cell : row ){
      printf("%7.3f\t", cell);
    }
    printf("\n");
  }

    vector<float> vf = GetActionValueEstimate( f_rewards, f_probabilities, );

    return (int) ( std::max_element(vf.begin(), vf.end() ) - vf.begin() );

}*/
template <class T>
boost::python::list toPythonList(std::vector<T> vector) {
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

template<typename T>
vector< vector< vector<T> > > convert3 (numeric::array a){

  vector< vector< vector<T> > > out;
  int a_len = len(a);
  for( int i = 0; i < a_len; i ++ ){
    numeric::array k = extract<numeric::array>(a[i]);
    int k_len = len(k);
    vector< vector<T> > sub;

    for( int j = 0; j < k_len; j++ ) {
      numeric::array k2 = extract<numeric::array>(k[j]);
      int p_len = len(k2);
      vector<T> sub2;
      for( int j2 = 0; j2 < p_len; j2++ ){

        T t = static_cast<T>( extract<double>( k2[j2] ) );
        sub2.push_back(t);

      }
      sub.push_back(sub2);

    }

    out.push_back(sub);
  }

  return out;

}


// Functions to demonstrate extraction
boost::python::list value_iteration(numeric::array rewards, numeric::array probabilities) {

  // Access a built-in type (an array)
  //boost::python::numeric::array a = data;
  // Need to <extract> array elements because their type is unknown.
  //a = extract<boost::python::numeric::array>(a[0]);
  //return extract<int> ( a[2] );
  vector<vector<double> > f_rewards = convert<double>( rewards );
  //vector<vector<bool>> f_mask = convert<bool>( mask );

  vector<vector<vector<double> > > f_probs = convert3<double>( probabilities );
  //vector<vector<bool>> f_pmask = convert<bool>( pmask );

/*  for( auto row : f_pseudo ){
    for( auto cell : row ){
      printf("%7.3f\t", cell);
    }
    printf("\n");
  }*/

  vector<vector<double>> output = ValueIterate( f_rewards, f_probs );

  boost::python::list toplist;

  for(int i = 0; i < output.size(); i++ ){
    boost::python::list l1list = toPythonList<double>(output[i]);
    toplist.append(l1list);
  }

  //return (int) ( std::max_element(vf.begin(), vf.end() ) - vf.begin() );
  return toplist;

}
// Functions to demonstrate extraction
boost::python::list value_iteration_pr(numeric::array rewards, numeric::array probabilities, numeric::array pseudo) {

  // Access a built-in type (an array)
  //boost::python::numeric::array a = data;
  // Need to <extract> array elements because their type is unknown.
  //a = extract<boost::python::numeric::array>(a[0]);
  //return extract<int> ( a[2] );
  vector<vector<double> > f_rewards = convert<double>( rewards );
  //vector<vector<bool>> f_mask = convert<bool>( mask );

  vector<vector<vector<double> > > f_probs = convert3<double>( probabilities );
  //vector<vector<bool>> f_pmask = convert<bool>( pmask );

  vector<vector<double> > f_pseudo = convert<double>( pseudo );

/*  for( auto row : f_pseudo ){
    for( auto cell : row ){
      printf("%7.3f\t", cell);
    }
    printf("\n");
  }*/

  vector<vector<double>> output = ValueIterate_pR( f_rewards, f_probs, f_pseudo );

  boost::python::list toplist;

  for(int i = 0; i < output.size(); i++ ){
    boost::python::list l1list = toPythonList<double>(output[i]);
    toplist.append(l1list);
  }

  //return (int) ( std::max_element(vf.begin(), vf.end() ) - vf.begin() );
  return toplist;

}


// Expose classes and methods to Python
BOOST_PYTHON_MODULE(planner) {

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    //def("get_best_action", &get_best_action);
    def("value_iteration", &value_iteration);
    def("value_iteration_pr", &value_iteration_pr);
};
