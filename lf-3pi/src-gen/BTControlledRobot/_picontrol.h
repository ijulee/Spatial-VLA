#ifndef _PICONTROL_H
#define _PICONTROL_H
#include "include/core/reactor.h"
#ifndef TOP_LEVEL_PREAMBLE_635068762_H
#define TOP_LEVEL_PREAMBLE_635068762_H
#include <math.h>
#define WHEEL_DIAMETER 0.032 // meters
#define COUNTS_PER_REV 360 //CPR
#define TICKS_PER_METER (WHEEL_DIAMETER * M_PI) / COUNTS_PER_REV
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
} _picontrol_err_t;
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
} _picontrol_ctrl_t;
typedef struct {
    struct self_base_t base;
    #line 80 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    float p_gain;
    #line 80 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    float i_gain;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    float acum_err;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    _picontrol_err_t* _lf_err;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    // width of -2 indicates that it is not a multiport.
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    int _lf_err_width;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    // Default input (in case it does not get connected)
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    _picontrol_err_t _lf_default__err;
    #line 82 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    _picontrol_ctrl_t _lf_ctrl;
    #line 82 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    int _lf_ctrl_width;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    reaction_t _lf__reaction_0;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    trigger_t _lf__err;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    reaction_t* _lf__err_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
} _picontrol_self_t;
_picontrol_self_t* new__picontrol();
#endif // _PICONTROL_H
