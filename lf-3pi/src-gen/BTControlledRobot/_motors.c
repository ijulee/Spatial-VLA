#include "include/api/api.h"
#include "include/Motors/Motors.h"
#include "_motors.h"
// ***** Start of method declarations.
void _motors_method_set_power(void* instance_args, float power, bool forward, bool left);
#define set_power(...) _motors_method_set_power(self, ##__VA_ARGS__)
#line 32 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
// Implementation of method set_power()
void _motors_method_set_power(void* instance_args, float power, bool forward, bool left) {
    _motors_self_t* self = (_motors_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    #line 33 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    power = fabsf(power);
    if (power > 1.0f) power = 1.0f;
    // Round down:
    uint16_t duty_cycle = (uint16_t)(power * MAX_MOTOR_POWER);
    motors_set_power(duty_cycle, forward, left);
}
#undef set_power
// ***** End of method declarations.
#include "include/api/set.h"
#define set_power(...) _motors_method_set_power(self, ##__VA_ARGS__)
void _motorsreaction_function_0(void* instance_args) {
    _motors_self_t* self = (_motors_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    
    #line 29 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    motors_init(1); // 1 indicates no clock division.
}
#undef set_power
#include "include/api/set_undef.h"
#include "include/api/set.h"
#define set_power(...) _motors_method_set_power(self, ##__VA_ARGS__)
void _motorsreaction_function_1(void* instance_args) {
    _motors_self_t* self = (_motors_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    _motors_left_power_t* left_power = self->_lf_left_power;
    int left_power_width = self->_lf_left_power_width; SUPPRESS_UNUSED_WARNING(left_power_width);
    #line 41 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    bool forward = (left_power->value >= 0.0f);
    set_power(left_power->value, forward, true);
}
#undef set_power
#include "include/api/set_undef.h"
#include "include/api/set.h"
#define set_power(...) _motors_method_set_power(self, ##__VA_ARGS__)
void _motorsreaction_function_2(void* instance_args) {
    _motors_self_t* self = (_motors_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    _motors_right_power_t* right_power = self->_lf_right_power;
    int right_power_width = self->_lf_right_power_width; SUPPRESS_UNUSED_WARNING(right_power_width);
    #line 46 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    bool forward = (right_power->value >= 0.0f);
    set_power(right_power->value, forward, false);
}
#undef set_power
#include "include/api/set_undef.h"
_motors_self_t* new__motors() {
    _motors_self_t* self = (_motors_self_t*)_lf_new_reactor(sizeof(_motors_self_t));
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Set input by default to an always absent default input.
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf_left_power = &self->_lf_default__left_power;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Set the default source reactor pointer
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf_default__left_power._base.source_reactor = (self_base_t*)self;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Set input by default to an always absent default input.
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf_right_power = &self->_lf_default__right_power;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    // Set the default source reactor pointer
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf_default__right_power._base.source_reactor = (self_base_t*)self;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.number = 0;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.function = _motorsreaction_function_0;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.self = self;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.deadline_violation_handler = NULL;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.STP_handler = NULL;
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.name = "?";
    #line 28 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_0.mode = NULL;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.number = 1;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.function = _motorsreaction_function_1;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.self = self;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.deadline_violation_handler = NULL;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.STP_handler = NULL;
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.name = "?";
    #line 40 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_1.mode = NULL;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.number = 2;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.function = _motorsreaction_function_2;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.self = self;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.deadline_violation_handler = NULL;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.STP_handler = NULL;
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.name = "?";
    #line 45 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__reaction_2.mode = NULL;
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__startup.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    self->_lf__startup_reactions[0] = &self->_lf__reaction_0;
    self->_lf__startup.last = NULL;
    self->_lf__startup.reactions = &self->_lf__startup_reactions[0];
    self->_lf__startup.number_of_reactions = 1;
    self->_lf__startup.is_timer = false;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power.last = NULL;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power_reactions[0] = &self->_lf__reaction_1;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power.reactions = &self->_lf__left_power_reactions[0];
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power.number_of_reactions = 1;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #ifdef FEDERATED
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__left_power.physical_time_of_arrival = NEVER;
    #line 25 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #endif // FEDERATED
    self->_lf__left_power.tmplt.type.element_size = sizeof(float);
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power.last = NULL;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power_reactions[0] = &self->_lf__reaction_2;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power.reactions = &self->_lf__right_power_reactions[0];
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power.number_of_reactions = 1;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #ifdef FEDERATED
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    self->_lf__right_power.physical_time_of_arrival = NEVER;
    #line 26 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/Motors.lf"
    #endif // FEDERATED
    self->_lf__right_power.tmplt.type.element_size = sizeof(float);
    return self;
}
