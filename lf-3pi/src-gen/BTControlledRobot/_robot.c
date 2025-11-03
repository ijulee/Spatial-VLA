#include "include/api/api.h"
#include "include/BTControlledRobot/Robot.h"
#include "_robot.h"
// ***** Start of method declarations.
// ***** End of method declarations.
#include "include/api/set.h"
void _robotreaction_function_0(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    
    #line 68 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->startAngle = 0.0f;
    self->startDistL = 0.0f;
    self->startDistR = 0.0f;
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_1(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct encoder {
        _encoders_trigger_t* trigger;
    
    } encoder;
    struct gyro {
        _gyroangle_trigger_t* trigger;
    
    } gyro;
    encoder.trigger = &(self->_lf_encoder.trigger);
    gyro.trigger = &(self->_lf_gyro.trigger);
    #line 74 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    /*, accel.trigger*/ 
            // update distance
            lf_set(encoder.trigger, true);
    
            // update angle
            lf_set(gyro.trigger, true);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_2(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct motor {
        _motorswithfeedback_left_speed_t* left_speed;
    _motorswithfeedback_right_speed_t* right_speed;
    
    } motor;
    _robot_command_t* command = self->_lf_command;
    int command_width = self->_lf_command_width; SUPPRESS_UNUSED_WARNING(command_width);
    _robot_notify0_t* notify0 = &self->_lf_notify0;
    motor.left_speed = &(self->_lf_motor.left_speed);
    motor.right_speed = &(self->_lf_motor.right_speed);
    reactor_mode_t* FORWARD = &self->_lf__modes[1];
    lf_mode_change_type_t _lf_FORWARD_change_type = reset_transition;
    reactor_mode_t* BACK = &self->_lf__modes[2];
    lf_mode_change_type_t _lf_BACK_change_type = reset_transition;
    reactor_mode_t* RIGHT = &self->_lf__modes[4];
    lf_mode_change_type_t _lf_RIGHT_change_type = reset_transition;
    reactor_mode_t* LEFT = &self->_lf__modes[3];
    lf_mode_change_type_t _lf_LEFT_change_type = reset_transition;
    reactor_mode_t* WAIT = &self->_lf__modes[0];
    lf_mode_change_type_t _lf_WAIT_change_type = reset_transition;
    #line 88 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    string next_command;
    
    if(command->is_present) {
        next_command = command->value;
    } else { // timed out, continue last command
        next_command = self->lastCommand;
    }
    
    self->lastCommand = next_command;
    char buf[17];
    sprintf(buf, "cmd: %s", next_command);
    lf_set(notify0,  next_command);
    
    switch (next_command[0]) {
        case 'f':
        lf_set(motor.left_speed, 0.1f);
        lf_set(motor.right_speed, 0.1f);
        lf_set_mode(FORWARD);
        break;
    
        case 'b':
        lf_set(motor.left_speed, -0.1f);
        lf_set(motor.right_speed, -0.1f);
        lf_set_mode(BACK);
        break;
    
        case 'l':
        lf_set(motor.left_speed, -0.05f);
        lf_set(motor.right_speed, 0.05f);
        lf_set_mode(LEFT);
        break;
    
        case 'r':
        lf_set(motor.left_speed, 0.05f);
        lf_set(motor.right_speed, -0.05f);
        lf_set_mode(RIGHT);
        break;
    
        case 's':
        lf_set(motor.left_speed, 0.0f);
        lf_set(motor.right_speed, 0.0f);
        lf_set_mode(WAIT);
        break;
    
        default:
        return;
    }
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_3(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct gyro {
        _gyroangle_z_t* z;
    
    } gyro;
    struct dist {
        _encoderdist_left_t* left;
    _encoderdist_right_t* right;
    
    } dist;
    gyro.z = self->_lf_gyro.z;
    dist.left = self->_lf_dist.left;
    dist.right = self->_lf_dist.right;
    #line 140 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->startAngle = gyro.z->value;
    self->startDistL = dist.left->value;
    self->startDistR = dist.right->value;
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_4(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct gyro {
        _gyroangle_z_t* z;
    
    } gyro;
    struct motor {
        _motorswithfeedback_left_speed_t* left_speed;
    _motorswithfeedback_right_speed_t* right_speed;
    
    } motor;
    gyro.z = self->_lf_gyro.z;
    _robot_notify3_t* notify3 = &self->_lf_notify3;
    motor.left_speed = &(self->_lf_motor.left_speed);
    motor.right_speed = &(self->_lf_motor.right_speed);
    #line 151 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float angleTurned = gyro.z->value - self->startAngle;
    char buf[17];
    sprintf(buf, "%3.2f", angleTurned);
    lf_set(notify3, buf);
    lf_set(motor.left_speed, 0.1 * (1 - (angleTurned / 20)));
    lf_set(motor.right_speed, 0.1 * (1 + (angleTurned / 20)));
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_5(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct dist {
        _encoderdist_left_t* left;
    _encoderdist_right_t* right;
    
    } dist;
    dist.left = self->_lf_dist.left;
    dist.right = self->_lf_dist.right;
    _robot_notify1_t* notify1 = &self->_lf_notify1;
    _robot_notify2_t* notify2 = &self->_lf_notify2;
    #line 162 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float distMovedL = dist.left->value - self->startDistL;
    float distMovedR = dist.right->value - self->startDistR;
    char buf1[17], buf2[17];
    sprintf(buf1, "%3.2f", distMovedL);
    sprintf(buf2, "%3.2f", distMovedR);
    lf_set(notify1, buf1);
    lf_set(notify2, buf2);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_6(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    reactor_mode_t* WAIT = &self->_lf__modes[0];
    lf_mode_change_type_t _lf_WAIT_change_type = reset_transition;
    #line 174 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set_mode(WAIT);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_7(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct gyro {
        _gyroangle_z_t* z;
    
    } gyro;
    struct motor {
        _motorswithfeedback_left_speed_t* left_speed;
    _motorswithfeedback_right_speed_t* right_speed;
    
    } motor;
    gyro.z = self->_lf_gyro.z;
    _robot_notify3_t* notify3 = &self->_lf_notify3;
    motor.left_speed = &(self->_lf_motor.left_speed);
    motor.right_speed = &(self->_lf_motor.right_speed);
    #line 183 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float angleTurned = gyro.z->value - self->startAngle;
    char buf[17];
    sprintf(buf, "%3.2f", angleTurned);
    lf_set(notify3, buf);
    lf_set(motor.left_speed, -0.1 * (1 + (angleTurned / 20)));
    lf_set(motor.right_speed, -0.1 * (1 - (angleTurned / 20)));
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_8(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct dist {
        _encoderdist_left_t* left;
    _encoderdist_right_t* right;
    
    } dist;
    dist.left = self->_lf_dist.left;
    dist.right = self->_lf_dist.right;
    _robot_notify1_t* notify1 = &self->_lf_notify1;
    _robot_notify2_t* notify2 = &self->_lf_notify2;
    #line 194 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float distMovedL = dist.left->value - self->startDistL;
    float distMovedR = dist.right->value - self->startDistR;
    char buf1[17], buf2[17];
    sprintf(buf1, "%3.2f", distMovedL);
    sprintf(buf2, "%3.2f", distMovedR);
    lf_set(notify1, buf1);
    lf_set(notify2, buf2);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_9(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    reactor_mode_t* WAIT = &self->_lf__modes[0];
    lf_mode_change_type_t _lf_WAIT_change_type = reset_transition;
    #line 206 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set_mode(WAIT);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_10(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct gyro {
        _gyroangle_z_t* z;
    
    } gyro;
    gyro.z = self->_lf_gyro.z;
    _robot_notify3_t* notify3 = &self->_lf_notify3;
    #line 215 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float angleTurned = gyro.z->value - self->startAngle;
    char buf[17];
    sprintf(buf, "%3.2f", angleTurned);
    lf_set(notify3, buf);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_11(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct dist {
        _encoderdist_left_t* left;
    _encoderdist_right_t* right;
    
    } dist;
    struct motor {
        _motorswithfeedback_left_speed_t* left_speed;
    _motorswithfeedback_right_speed_t* right_speed;
    
    } motor;
    dist.left = self->_lf_dist.left;
    dist.right = self->_lf_dist.right;
    _robot_notify1_t* notify1 = &self->_lf_notify1;
    _robot_notify2_t* notify2 = &self->_lf_notify2;
    motor.left_speed = &(self->_lf_motor.left_speed);
    motor.right_speed = &(self->_lf_motor.right_speed);
    #line 225 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float distMovedL = dist.left->value - self->startDistL;
    float distMovedR = dist.right->value - self->startDistR;
    char buf1[17], buf2[17];
    sprintf(buf1, "%3.2f", distMovedL);
    sprintf(buf2, "%3.2f", distMovedR);
    lf_set(notify1, buf1);
    lf_set(notify2, buf2);
    
    lf_set(motor.left_speed, -0.1 * (1 + ((distMovedR+distMovedL) / 20)));
    lf_set(motor.right_speed, 0.1 * (1 - ((distMovedR+distMovedL) / 20)));
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_12(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    reactor_mode_t* WAIT = &self->_lf__modes[0];
    lf_mode_change_type_t _lf_WAIT_change_type = reset_transition;
    #line 240 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set_mode(WAIT);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_13(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct gyro {
        _gyroangle_z_t* z;
    
    } gyro;
    gyro.z = self->_lf_gyro.z;
    _robot_notify3_t* notify3 = &self->_lf_notify3;
    #line 249 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float angleTurned = gyro.z->value - self->startAngle;
    char buf[17];
    sprintf(buf, "%3.2f", angleTurned);
    lf_set(notify3, buf);
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_14(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct dist {
        _encoderdist_left_t* left;
    _encoderdist_right_t* right;
    
    } dist;
    struct motor {
        _motorswithfeedback_left_speed_t* left_speed;
    _motorswithfeedback_right_speed_t* right_speed;
    
    } motor;
    dist.left = self->_lf_dist.left;
    dist.right = self->_lf_dist.right;
    _robot_notify1_t* notify1 = &self->_lf_notify1;
    _robot_notify2_t* notify2 = &self->_lf_notify2;
    motor.left_speed = &(self->_lf_motor.left_speed);
    motor.right_speed = &(self->_lf_motor.right_speed);
    #line 259 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    float distMovedL = dist.left->value - self->startDistL;
    float distMovedR = dist.right->value - self->startDistR;
    char buf1[17], buf2[17];
    sprintf(buf1, "%3.2f", distMovedL);
    sprintf(buf2, "%3.2f", distMovedR);
    lf_set(notify1, buf1);
    lf_set(notify2, buf2);
    
    lf_set(motor.left_speed, 0.1 * (1 + ((distMovedR+distMovedL) / 20)));
    lf_set(motor.right_speed, -0.1 * (1 - ((distMovedR+distMovedL) / 20)));
}
#include "include/api/set_undef.h"
#include "include/api/set.h"
void _robotreaction_function_15(void* instance_args) {
    _robot_self_t* self = (_robot_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    reactor_mode_t* WAIT = &self->_lf__modes[0];
    lf_mode_change_type_t _lf_WAIT_change_type = reset_transition;
    #line 274 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set_mode(WAIT);
}
#include "include/api/set_undef.h"
_robot_self_t* new__robot() {
    _robot_self_t* self = (_robot_self_t*)_lf_new_reactor(sizeof(_robot_self_t));
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set input by default to an always absent default input.
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_command = &self->_lf_default__command;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set the default source reactor pointer
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_default__command._base.source_reactor = (self_base_t*)self;
    // Set the _width variable for all cases. This will be -2
    // if the reactor is not a bank of reactors.
    self->_lf_encoder_width = -2;
    // Set the _width variable for all cases. This will be -2
    // if the reactor is not a bank of reactors.
    self->_lf_gyro_width = -2;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_trigger.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_reactions[0] = &self->_lf__reaction_3;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_reactions[1] = &self->_lf__reaction_4;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_reactions[2] = &self->_lf__reaction_7;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_reactions[3] = &self->_lf__reaction_10;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_reactions[4] = &self->_lf__reaction_13;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_trigger.reactions = self->_lf_gyro.z_reactions;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_trigger.last = NULL;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_trigger.number_of_reactions = 5;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    #ifdef FEDERATED
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    self->_lf_gyro.z_trigger.physical_time_of_arrival = NEVER;
    #line 122 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/IMU.lf"
    #endif // FEDERATED
    // Set the _width variable for all cases. This will be -2
    // if the reactor is not a bank of reactors.
    self->_lf_motor_width = -2;
    // Set the _width variable for all cases. This will be -2
    // if the reactor is not a bank of reactors.
    self->_lf_dist_width = -2;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_trigger.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_reactions[0] = &self->_lf__reaction_3;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_reactions[1] = &self->_lf__reaction_5;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_reactions[2] = &self->_lf__reaction_8;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_reactions[3] = &self->_lf__reaction_11;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_reactions[4] = &self->_lf__reaction_14;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_trigger.reactions = self->_lf_dist.left_reactions;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_trigger.last = NULL;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_trigger.number_of_reactions = 5;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.left_trigger.physical_time_of_arrival = NEVER;
    #line 23 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_trigger.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_reactions[0] = &self->_lf__reaction_3;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_reactions[1] = &self->_lf__reaction_5;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_reactions[2] = &self->_lf__reaction_8;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_reactions[3] = &self->_lf__reaction_11;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_reactions[4] = &self->_lf__reaction_14;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_trigger.reactions = self->_lf_dist.right_reactions;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_trigger.last = NULL;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_trigger.number_of_reactions = 5;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_dist.right_trigger.physical_time_of_arrival = NEVER;
    #line 24 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.number = 0;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.function = _robotreaction_function_0;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.self = self;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.deadline_violation_handler = NULL;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.STP_handler = NULL;
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.name = "?";
    #line 67 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.mode = NULL;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.number = 1;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.function = _robotreaction_function_1;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.self = self;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.deadline_violation_handler = NULL;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.STP_handler = NULL;
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.name = "?";
    #line 73 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_1.mode = NULL;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.number = 2;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.function = _robotreaction_function_2;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.self = self;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.deadline_violation_handler = NULL;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.STP_handler = NULL;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.name = "?";
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_2.mode = &self->_lf__modes[0];
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.number = 3;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.function = _robotreaction_function_3;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.self = self;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.deadline_violation_handler = NULL;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.STP_handler = NULL;
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.name = "?";
    #line 139 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_3.mode = &self->_lf__modes[0];
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.number = 4;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.function = _robotreaction_function_4;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.self = self;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.deadline_violation_handler = NULL;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.STP_handler = NULL;
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.name = "?";
    #line 150 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_4.mode = &self->_lf__modes[1];
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.number = 5;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.function = _robotreaction_function_5;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.self = self;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.deadline_violation_handler = NULL;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.STP_handler = NULL;
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.name = "?";
    #line 161 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_5.mode = &self->_lf__modes[1];
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.number = 6;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.function = _robotreaction_function_6;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.self = self;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.deadline_violation_handler = NULL;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.STP_handler = NULL;
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.name = "?";
    #line 173 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_6.mode = &self->_lf__modes[1];
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.number = 7;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.function = _robotreaction_function_7;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.self = self;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.deadline_violation_handler = NULL;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.STP_handler = NULL;
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.name = "?";
    #line 182 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_7.mode = &self->_lf__modes[2];
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.number = 8;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.function = _robotreaction_function_8;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.self = self;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.deadline_violation_handler = NULL;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.STP_handler = NULL;
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.name = "?";
    #line 193 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_8.mode = &self->_lf__modes[2];
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.number = 9;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.function = _robotreaction_function_9;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.self = self;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.deadline_violation_handler = NULL;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.STP_handler = NULL;
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.name = "?";
    #line 205 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_9.mode = &self->_lf__modes[2];
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.number = 10;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.function = _robotreaction_function_10;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.self = self;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.deadline_violation_handler = NULL;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.STP_handler = NULL;
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.name = "?";
    #line 214 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_10.mode = &self->_lf__modes[3];
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.number = 11;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.function = _robotreaction_function_11;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.self = self;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.deadline_violation_handler = NULL;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.STP_handler = NULL;
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.name = "?";
    #line 223 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_11.mode = &self->_lf__modes[3];
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.number = 12;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.function = _robotreaction_function_12;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.self = self;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.deadline_violation_handler = NULL;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.STP_handler = NULL;
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.name = "?";
    #line 239 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_12.mode = &self->_lf__modes[3];
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.number = 13;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.function = _robotreaction_function_13;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.self = self;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.deadline_violation_handler = NULL;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.STP_handler = NULL;
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.name = "?";
    #line 248 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_13.mode = &self->_lf__modes[4];
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.number = 14;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.function = _robotreaction_function_14;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.self = self;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.deadline_violation_handler = NULL;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.STP_handler = NULL;
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.name = "?";
    #line 257 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_14.mode = &self->_lf__modes[4];
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.number = 15;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.function = _robotreaction_function_15;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.self = self;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.deadline_violation_handler = NULL;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.STP_handler = NULL;
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.name = "?";
    #line 273 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_15.mode = &self->_lf__modes[4];
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling.last = NULL;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling_reactions[0] = &self->_lf__reaction_1;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling.reactions = &self->_lf__sampling_reactions[0];
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling.number_of_reactions = 1;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__sampling.physical_time_of_arrival = NEVER;
    #line 56 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__sampling.is_timer = true;
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__sampling.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command.last = NULL;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command_reactions[0] = &self->_lf__reaction_6;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command_reactions[1] = &self->_lf__reaction_9;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command_reactions[2] = &self->_lf__reaction_12;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command_reactions[3] = &self->_lf__reaction_15;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command.reactions = &self->_lf__get_command_reactions[0];
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command.number_of_reactions = 4;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__get_command.physical_time_of_arrival = NEVER;
    #line 57 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__get_command.is_timer = true;
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__get_command.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout.last = NULL;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout_reactions[0] = &self->_lf__reaction_2;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout.reactions = &self->_lf__timeout_reactions[0];
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout.number_of_reactions = 1;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__timeout.physical_time_of_arrival = NEVER;
    #line 83 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__timeout.is_timer = true;
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__timeout.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__startup.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    self->_lf__startup_reactions[0] = &self->_lf__reaction_0;
    self->_lf__startup.last = NULL;
    self->_lf__startup.reactions = &self->_lf__startup_reactions[0];
    self->_lf__startup.number_of_reactions = 1;
    self->_lf__startup.is_timer = false;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command.last = NULL;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command_reactions[0] = &self->_lf__reaction_2;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command.reactions = &self->_lf__command_reactions[0];
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command.number_of_reactions = 1;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__command.physical_time_of_arrival = NEVER;
    #line 38 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__command.tmplt.type.element_size = sizeof(string);
    // Initialize modes
    self_base_t* _lf_self_base = (self_base_t*)self;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[0].state = &_lf_self_base->_lf__mode_state;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[0].name = "WAIT";
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[0].deactivation_time = 0;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[0].flags = 0;
    #line 148 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[1].state = &_lf_self_base->_lf__mode_state;
    #line 148 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[1].name = "FORWARD";
    #line 148 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[1].deactivation_time = 0;
    #line 148 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[1].flags = 0;
    #line 180 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[2].state = &_lf_self_base->_lf__mode_state;
    #line 180 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[2].name = "BACK";
    #line 180 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[2].deactivation_time = 0;
    #line 180 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[2].flags = 0;
    #line 212 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[3].state = &_lf_self_base->_lf__mode_state;
    #line 212 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[3].name = "LEFT";
    #line 212 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[3].deactivation_time = 0;
    #line 212 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[3].flags = 0;
    #line 246 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[4].state = &_lf_self_base->_lf__mode_state;
    #line 246 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[4].name = "RIGHT";
    #line 246 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[4].deactivation_time = 0;
    #line 246 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__modes[4].flags = 0;
    // Initialize mode state
    _lf_self_base->_lf__mode_state.parent_mode = NULL;
    _lf_self_base->_lf__mode_state.initial_mode = &self->_lf__modes[0];
    _lf_self_base->_lf__mode_state.current_mode = _lf_self_base->_lf__mode_state.initial_mode;
    _lf_self_base->_lf__mode_state.next_mode = NULL;
    _lf_self_base->_lf__mode_state.mode_change = no_transition;
    return self;
}
