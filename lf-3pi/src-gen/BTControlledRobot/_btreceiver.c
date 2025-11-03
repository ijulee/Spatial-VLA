#include "include/api/api.h"
#include "include/BTControlledRobot/BTReceiver.h"
#include "_btreceiver.h"
// ***** Start of method declarations.
// ***** End of method declarations.
_btreceiver_self_t* new__btreceiver() {
    _btreceiver_self_t* self = (_btreceiver_self_t*)_lf_new_reactor(sizeof(_btreceiver_self_t));
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set input by default to an always absent default input.
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_enable = &self->_lf_default__enable;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    // Set the default source reactor pointer
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf_default__enable._base.source_reactor = (self_base_t*)self;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__enable.last = NULL;
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #ifdef FEDERATED_DECENTRALIZED
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    self->_lf__enable.intended_tag = (tag_t) { .time = NEVER, .microstep = 0u};
    #line 282 "/home/foobar/Spatial-VLA/lf-3pi/src/BTControlledRobot.lf"
    #endif // FEDERATED_DECENTRALIZED
    self->_lf__enable.tmplt.type.element_size = sizeof(bool);
    return self;
}
