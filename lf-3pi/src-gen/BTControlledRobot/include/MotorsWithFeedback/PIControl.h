#ifndef _picontrol_H
#define _picontrol_H
#ifndef TOP_LEVEL_PREAMBLE_635068762_H
#define TOP_LEVEL_PREAMBLE_635068762_H
/*Correspondence: Range: [(21, 2), (24, 64)) -> Range: [(0, 0), (3, 64)) (verbatim=true; src=/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf)*/#include <math.h>
#define WHEEL_DIAMETER 0.032 // meters
#define COUNTS_PER_REV 360 //CPR
#define TICKS_PER_METER (WHEEL_DIAMETER * M_PI) / COUNTS_PER_REV
#endif
#ifdef __cplusplus
extern "C" {
#endif
#include "../include/api/api.h"
#include "../include/core/reactor.h"
#ifdef __cplusplus
}
#endif
typedef struct picontrol_self_t{
    self_base_t base; // This field is only to be used by the runtime, not the user.
    float p_gain;
    float i_gain;
    float acum_err;
    int end[0]; // placeholder; MSVC does not compile empty structs
} picontrol_self_t;
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
} picontrol_err_t;
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
} picontrol_ctrl_t;
#endif
