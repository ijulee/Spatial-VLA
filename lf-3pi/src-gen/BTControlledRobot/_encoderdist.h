#ifndef _ENCODERDIST_H
#define _ENCODERDIST_H
#include "include/core/reactor.h"
#ifndef TOP_LEVEL_PREAMBLE_81505591_H
#define TOP_LEVEL_PREAMBLE_81505591_H
#include <hardware/gpio.h>
#include <pico/stdlib.h>
#endif
typedef struct {
    token_type_t type;
    lf_token_t* token;
    size_t length;
    bool is_present;
    lf_port_internal_t _base;
    int32_t value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _encoderdist_angle_left_t;
typedef struct {
    token_type_t type;
    lf_token_t* token;
    size_t length;
    bool is_present;
    lf_port_internal_t _base;
    int32_t value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _encoderdist_angle_right_t;
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
} _encoderdist_left_t;
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
} _encoderdist_right_t;
typedef struct {
    struct self_base_t base;
    
    
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_angle_left_t* _lf_angle_left;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // width of -2 indicates that it is not a multiport.
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_angle_left_width;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Default input (in case it does not get connected)
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_angle_left_t _lf_default__angle_left;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_angle_right_t* _lf_angle_right;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // width of -2 indicates that it is not a multiport.
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_angle_right_width;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Default input (in case it does not get connected)
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_angle_right_t _lf_default__angle_right;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_left_t _lf_left;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_left_width;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _encoderdist_right_t _lf_right;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_right_width;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_0;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__angle_left;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__angle_left_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__angle_right;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__angle_right_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
} _encoderdist_self_t;
_encoderdist_self_t* new__encoderdist();
#endif // _ENCODERDIST_H
