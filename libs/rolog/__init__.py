# USAGE:
# Choose a logging level for the tracking library. Use MODE_STD for stdout.
# import rolog
# roslog.config(
#                   Can also be MODE_DISABLED or MODE_STD
#     rolog.Profile(rolog.Profile.MODE_ROS2,
#                         node=tracker_node))
# rolog.enable_sessions([session_name, ...])
# rolog.log('INFO', session_name, message)

from .session_log import \
    Profile, \
    config, \
    enable_sessions, \
    is_session_enabled, \
    log, info, error, warn, debug

from .profiler import stat, show_profile, cycle_profile
