#ifndef _BTRECEIVER_H
#define _BTRECEIVER_H
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
    bool value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _btreceiver_enable_t;
typedef struct {
    token_type_t type;
    lf_token_t* token;
    size_t length;
    bool is_present;
    lf_port_internal_t _base;
    string value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _btreceiver_command_t;
typedef struct {
    struct self_base_t base;
    
    
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _btreceiver_enable_t* _lf_enable;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // width of -2 indicates that it is not a multiport.
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_enable_width;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Default input (in case it does not get connected)
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _btreceiver_enable_t _lf_default__enable;
    #line 283 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _btreceiver_command_t _lf_command;
    #line 283 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_command_width;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__enable;
    #ifdef FEDERATED
    
    #endif // FEDERATED
} _btreceiver_self_t;
_btreceiver_self_t* new__btreceiver();
#endif // _BTRECEIVER_H
