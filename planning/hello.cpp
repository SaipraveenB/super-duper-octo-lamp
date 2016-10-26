//  Copyright Joel de Guzman 2002-2004. Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file LICENSE_1_0.txt
//  or copy at http://www.boost.org/LICENSE_1_0.txt)
//  Hello World Example from the tutorial
//  [Joel de Guzman 10/9/2002]

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>

#include <vector>
#include <utility>

#include <boost/python/tuple.py>

#include "planner.h"

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

template<typename T>
vector<vector<T>> convert(numeric::array a){

    vector<vector<T>> out;
    int a_len = len(a);
    int k_len = len(k);
    for( int i = 0; i < a_len; i ++ ){
        numeric::array k = extract<numeric::array>(a[i]);
        vector<T> sub;
        for( int j = 0; j < k_len; j++ )
            sub.push_back( extract<T>(k[j]) );
        out.push_back(sub);
    }

    return out;

}

using namespace boost::python;
// Functions to demonstrate extraction
int get_best_action(numeric::array rewards, numeric::array pseudo, tuple kernel_size, tuple start_state, int num_trajectories, double gamma, double learning_rate ) {

    // Access a built-in type (an array)
    //boost::python::numeric::array a = data;
    // Need to <extract> array elements because their type is unknown.
    //a = extract<boost::python::numeric::array>(a[0]);
    //return extract<int> ( a[2] );
    vector<vector<float>> f_rewards = convert<float>( rewards );
    //vector<vector<bool>> f_mask = convert<bool>( mask );

    vector<vector<float>> f_pseudo = convert<float>( rewards );
    //vector<vector<bool>> f_pmask = convert<bool>( pmask );

    pair<int,int> p_kern = make_pair( extract<int>(kernel_size[0]), extract<int>(kernel_size[1]) );
    pair<int,int> p_ss = make_pair( extract<int>(start_state[0]), extract<int>(start_state[1]) );

    GetActionValueEstimate( f_rewards, f_pseudo, p_ss, std::make_pair(-1,-1), p_kern );

}

// Functions to demonstrate extraction
numeric::array get_value_function_board(numeric::array rewards, numeric::array probability, tuple kernel_size, tuple start_state, int num_trajectories, double gamma, double learning_rate ) {

    // Access a built-in type (an array)
    //boost::python::numeric::array a = data;
    // Need to <extract> array elements because their type is unknown.
    //a = extract<boost::python::numeric::array>(a[0]);
    //return extract<int> ( a[2] );
    vector<vector<float>> f_rewards = convert<float>( rewards );
    //vector<vector<bool>> f_mask = convert<bool>( mask );

    vector<vector<float>> f_pseudo = convert<float>( rewards );
    //vector<vector<bool>> f_pmask = convert<bool>( pmask );

    pair<int,int> p_kern = make_pair( extract<int>(kernel_size[0]), extract<int>(kernel_size[1]) );
    pair<int,int> p_ss = make_pair( extract<int>(start_state[0]), extract<int>(start_state[1]) );

    ValueIterate( f_rewards, f_pseudo, p_ss, std::make_pair(-1,-1), p_kern );


}


// Expose classes and methods to Python
BOOST_PYTHON_MODULE(planner) {
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");


    def("setArray", &setArray);

}
