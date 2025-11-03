#include "include/api/api.h"
#include "include/BTControlledRobot/EncoderDist.h"
#include "_encoderdist.h"
// *********** From the preamble, verbatim:
#line 27 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
#def PI   3.14159
#def DIA  3.175

// *********** End of preamble.
// ***** Start of method declarations.
// ***** End of method declarations.
#include "include/api/set.h"
void _encoderdistreaction_function_0(void* instance_args) {
    _encoderdist_self_t* self = (_encoderdist_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    _encoderdist_angle_left_t* angle_left = self->_lf_angle_left;
    int angle_left_width = self->_lf_angle_left_width; SUPPRESS_UNUSED_WARNING(angle_left_width);
    _encoderdist_angle_right_t* angle_right = self->_lf_angle_right;
    int angle_right_width = self->_lf_angle_right_width; SUPPRESS_UNUSED_WARNING(angle_right_width);
    _encoderdist_left_t* left = &self->_lf_left;
    _encoderdist_right_t* right = &self->_lf_right;
    #line 32 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set(left, (angle_left->value/360.0f) * PI * DIA);
    lf_set(right, (angle_right->value/360.0f) * PI * DIA);
}
#include "include/api/set_undef.h"
_encoderdist_self_t* new__encoderdist() {
    _encoderdist_self_t* self = (_encoderdist_self_t*)_lf_new_reactor(sizeof(_encoderdist_self_t));
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set input by default to an always absent default input.
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_angle_left = &self->_lf_default__angle_left;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set the default source reactor pointer
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_default__angle_left._base.source_reactor = (self_base_t*)self;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set input by default to an always absent default input.
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_angle_right = &self->_lf_default__angle_right;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set the default source reactor pointer
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_default__angle_right._base.source_reactor = (self_base_t*)self;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.number = 0;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.function = _encoderdistreaction_function_0;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.self = self;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.deadline_violation_handler = NULL;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.STP_handler = NULL;
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.name = "?";
    #line 31 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.mode = NULL;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left.last = NULL;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left_reactions[0] = &self->_lf__reaction_0;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left.reactions = &self->_lf__angle_left_reactions[0];
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left.number_of_reactions = 1;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_left.physical_time_of_arrival = NEVER;
    #line 20 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__angle_left.tmplt.type.element_size = sizeof(int32_t);
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right.last = NULL;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right_reactions[0] = &self->_lf__reaction_0;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right.reactions = &self->_lf__angle_right_reactions[0];
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right.number_of_reactions = 1;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__angle_right.physical_time_of_arrival = NEVER;
    #line 21 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED
    self->_lf__angle_right.tmplt.type.element_size = sizeof(int32_t);
    return self;
}
