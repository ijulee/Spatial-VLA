#ifndef _BTCONTROLLEDROBOT_MAIN_H
#define _BTCONTROLLEDROBOT_MAIN_H
#include "include/core/reactor.h"
#include "_display.h"
#include "_btreceiver.h"
#include "_robot.h"
#ifndef TOP_LEVEL_PREAMBLE_81505591_H
#define TOP_LEVEL_PREAMBLE_81505591_H
#include <pico/stdlib.h>
#include <display.h>        // Do not use "display.h". Doesn't work.
#include <hardware/gpio.h>
#include <pico/stdlib.h>
#endif
typedef struct {
    struct self_base_t base;
    
    
    struct {
        #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        _btreceiver_enable_t enable;
    } _lf_rcvr;
    int _lf_rcvr_width;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_0;
    trigger_t _lf__startup;
    reaction_t* _lf__startup_reactions[1];
} _btcontrolledrobot_main_main_self_t;
_btcontrolledrobot_main_main_self_t* new__btcontrolledrobot_main();
#endif // _BTCONTROLLEDROBOT_MAIN_H
