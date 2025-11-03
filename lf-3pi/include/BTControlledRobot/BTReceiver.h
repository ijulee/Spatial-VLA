#ifndef _btreceiver_H
#define _btreceiver_H
#ifndef TOP_LEVEL_PREAMBLE_81505591_H
#define TOP_LEVEL_PREAMBLE_81505591_H
/*Correspondence: Range: [(14, 4), (15, 24)) -> Range: [(0, 0), (1, 24)) (verbatim=true; src=/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf)*/#include <hardware/gpio.h>
#include <pico/stdlib.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif
#include "../include/api/api.h"
#include "../include/core/reactor.h"
#ifdef __cplusplus
}
#endif
typedef struct btreceiver_self_t{
    self_base_t base; // This field is only to be used by the runtime, not the user.
    int end[0]; // placeholder; MSVC does not compile empty structs
} btreceiver_self_t;
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
} btreceiver_enable_t;
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
} btreceiver_command_t;
#endif
