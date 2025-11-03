#ifndef _ROBOT_H
#define _ROBOT_H
#include "include/core/reactor.h"
#include "_encoders.h"
#include "_gyroangle.h"
#include "_motorswithfeedback.h"
#include "_encoderdist.h"
#ifndef TOP_LEVEL_PREAMBLE_81505591_H
#define TOP_LEVEL_PREAMBLE_81505591_H
#include <pico/stdlib.h>
#include <imu.h>
#include <math.h>
#include <hardware/pio.h>
#include <quadrature_encoder.pio.h>

// pin defines
#define RIGHT_ENCODER_AB 8
#define LEFT_ENCODER_AB 12
#define RIGHT_SM 0
#define LEFT_SM 1
#include <hardware/gpio.h>
#include <pico/stdlib.h>
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
    string value;
    #ifdef FEDERATED
    #ifdef FEDERATED_DECENTRALIZED
    tag_t intended_tag;
    #endif
    interval_t physical_time_of_arrival;
    #endif
} _robot_command_t;
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
} _robot_notify0_t;
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
} _robot_notify1_t;
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
} _robot_notify2_t;
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
} _robot_notify3_t;
typedef struct {
    struct self_base_t base;
    #line 37 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    interval_t calibration_time;
    #line 37 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    interval_t sample_period;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    string lastCommand;
    #line 41 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float startAngle;
    #line 42 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float startDistL;
    #line 43 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float startDistR;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_command_t* _lf_command;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // width of -2 indicates that it is not a multiport.
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_command_width;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Default input (in case it does not get connected)
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_command_t _lf_default__command;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_notify0_t _lf_notify0;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_notify0_width;
    #line 46 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_notify1_t _lf_notify1;
    #line 46 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_notify1_width;
    #line 47 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_notify2_t _lf_notify2;
    #line 47 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_notify2_width;
    #line 48 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    _robot_notify3_t _lf_notify3;
    #line 48 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    int _lf_notify3_width;
    struct {
        #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Encoders.lf"
        _encoders_trigger_t trigger;
    } _lf_encoder;
    int _lf_encoder_width;
    struct {
        #line 118 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
        _gyroangle_trigger_t trigger;
        #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
        _gyroangle_z_t* z;
        #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
        trigger_t z_trigger;
        #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
        reaction_t* z_reactions[5];
    } _lf_gyro;
    int _lf_gyro_width;
    struct {
        #line 29 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
        _motorswithfeedback_left_speed_t left_speed;
        #line 30 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
        _motorswithfeedback_right_speed_t right_speed;
    } _lf_motor;
    int _lf_motor_width;
    struct {
        #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        _encoderdist_left_t* left;
        #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        trigger_t left_trigger;
        #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        reaction_t* left_reactions[5];
        #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        _encoderdist_right_t* right;
        #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        trigger_t right_trigger;
        #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
        reaction_t* right_reactions[5];
    } _lf_dist;
    int _lf_dist_width;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_0;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_1;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_2;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_3;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_4;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_5;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_6;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_7;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_8;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_9;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_10;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_11;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_12;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_13;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_14;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t _lf__reaction_15;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__sampling;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__sampling_reactions[1];
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__get_command;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__get_command_reactions[4];
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__timeout;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__timeout_reactions[1];
    trigger_t _lf__startup;
    reaction_t* _lf__startup_reactions[1];
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    trigger_t _lf__command;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    reaction_t* _lf__command_reactions[1];
    #ifdef FEDERATED
    
    #endif // FEDERATED
    reactor_mode_t _lf__modes[5];
} _robot_self_t;
_robot_self_t* new__robot();
#endif // _ROBOT_H
