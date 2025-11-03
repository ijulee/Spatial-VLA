#include "include/api/api.h"
#include "include/BTControlledRobot/BTControlledRobot.h"
#include "_btcontrolledrobot_main.h"
// ***** Start of method declarations.
// ***** End of method declarations.
#include "include/api/set.h"
void _btcontrolledrobot_mainreaction_function_0(void* instance_args) {
    _btcontrolledrobot_main_main_self_t* self = (_btcontrolledrobot_main_main_self_t*)instance_args; SUPPRESS_UNUSED_WARNING(self);
    struct rcvr {
        _btreceiver_enable_t* enable;
    
    } rcvr;
    rcvr.enable = &(self->_lf_rcvr.enable);
    #line 299 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    lf_set(rcvr.enable, true);
}
#include "include/api/set_undef.h"
_btcontrolledrobot_main_main_self_t* new__btcontrolledrobot_main() {
    _btcontrolledrobot_main_main_self_t* self = (_btcontrolledrobot_main_main_self_t*)_lf_new_reactor(sizeof(_btcontrolledrobot_main_main_self_t));
    // Set the _width variable for all cases. This will be -2
    // if the reactor is not a bank of reactors.
    self->_lf_rcvr_width = -2;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.number = 0;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.function = _btcontrolledrobot_mainreaction_function_0;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.self = self;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.deadline_violation_handler = NULL;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.STP_handler = NULL;
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.name = "?";
    #line 298 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__reaction_0.mode = NULL;
    #ifdef FEDERATED_DECENTRALIZED
    self->_lf__startup.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #endif // FEDERATED_DECENTRALIZED
    self->_lf__startup_reactions[0] = &self->_lf__reaction_0;
    self->_lf__startup.last = NULL;
    self->_lf__startup.reactions = &self->_lf__startup_reactions[0];
    self->_lf__startup.number_of_reactions = 1;
    self->_lf__startup.is_timer = false;
    return self;
}
