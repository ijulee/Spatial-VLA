#ifndef _MOTORS_H
#define _MOTORS_H
#include "include/core/reactor.h"
#ifndef TOP_LEVEL_PREAMBLE_538292941_H
#define TOP_LEVEL_PREAMBLE_538292941_H
#include <motors.h>
#include <math.h>
#endif
typedef struct {
    token_type_t type;
    lf_token_t* token;
    size_t length;
    bool is_present;
    lf_port_internal_t _base;
    float value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _motors_left_power_t;
typedef struct {
    token_type_t type;
    lf_token_t* token;
    size_t length;
    bool is_present;
    lf_port_internal_t _base;
    float value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _motors_right_power_t;
typedef struct {
    struct self_base_t base;
    
    
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    _motors_left_power_t* _lf_left_power;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // width of -2 indicates that it is not a multiport.
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    int _lf_left_power_width;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Default input (in case it does not get connected)
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    _motors_left_power_t _lf_default__left_power;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    _motors_right_power_t* _lf_right_power;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // width of -2 indicates that it is not a multiport.
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    int _lf_right_power_width;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Default input (in case it does not get connected)
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    _motors_right_power_t _lf_default__right_power;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    reaction_t _lf__reaction_0;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    reaction_t _lf__reaction_1;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    reaction_t _lf__reaction_2;
    trigger_t _lf__startup;
    reaction_t* _lf__startup_reactions[1];
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    trigger_t _lf__left_power;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    reaction_t* _lf__left_power_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    trigger_t _lf__right_power;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    reaction_t* _lf__right_power_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
} _motors_self_t;
_motors_self_t* new__motors();
#endif // _MOTORS_H
