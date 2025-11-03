#include "include/api/api.h"
#include "include/MotorsWithFeedback/PIControl.h"
#include "_picontrol.h"
// ***** Start of method declarations.
// ***** End of method declarations.
#include "include/api/set.h"
void _picontrolreaction_function_0(void* instance_args) {
    _picontrol_self_t* self = (_picontrol_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    _picontrol_err_t* err = self->_lf_err;
    int err_width = self->_lf_err_width; SUPPRESS_UNUSED_WARNING(err_width);
    _picontrol_ctrl_t* ctrl = &self->_lf_ctrl;
    #line 86 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->acum_err += err->value;
    lf_set(ctrl, err->value * self->p_gain +
                (self->acum_err) * self->i_gain);
}
#include "include/api/set_undef.h"
_picontrol_self_t* new__picontrol() {
    _picontrol_self_t* self = (_picontrol_self_t*)_lf_new_reactor(sizeof(_picontrol_self_t));
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    // Set input by default to an always absent default input.
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf_err = &self->_lf_default__err;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    // Set the default source reactor pointer
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf_default__err._base.source_reactor = (self_base_t*)self;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.number = 0;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.function = _picontrolreaction_function_0;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.self = self;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.deadline_violation_handler = NULL;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.STP_handler = NULL;
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.name = "?";
    #line 85 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__reaction_0.mode = NULL;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err.last = NULL;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    #endif // FEDERATED_DECENTRALIZED
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err_reactions[0] = &self->_lf__reaction_0;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err.reactions = &self->_lf__err_reactions[0];
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err.number_of_reactions = 1;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    #ifdef FEDERATED
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    self->_lf__err.physical_time_of_arrival = NEVER;
    #line 81 "/home/foobar/Spatial-VLA/lf-3pi/src/lib/MotorsWithFeedback.lf"
    #endif // FEDERATED
    self->_lf__err.tmplt.type.element_size = sizeof(float);
    return self;
}
