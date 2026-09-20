"""Non-blocking shutdown for standalone Isaac Lab applications."""
from __future__ import annotations

import weakref


def install_nonblocking_stop_handler(sim) -> None:
    """Replace Lab's render-until-PLAY callback with a queued quit request.

    Closing a Kit window emits timeline STOP. Rendering inside that callback
    prevents the window-close event from completing. Reset's temporary STOP
    events must still be ignored, using Lab's existing suppression flag.
    Store the replacement in Lab's own handle so clear_instance unsubscribes it.
    """
    handle = getattr(sim, "_app_control_on_stop_handle", None)
    if handle is None:
        return

    import omni.timeline

    sim_ref = weakref.ref(sim)

    def request_quit(_event):
        context = sim_ref()
        if context is not None and not context._disable_app_control_on_stop_handle:
            # post_quit queues shutdown; never update/render/close inside STOP.
            context._app.post_quit(0)

    handle.unsubscribe()
    stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    sim._app_control_on_stop_handle = stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.STOP), request_quit, order=15,
    )
