import numpy as np
from .colors import bcolors
from .code_inspect import get_method_name
np.set_printoptions(precision=2)

# g_mode can be one of:
# LOG_NONE: disabled
# LOG_ROS2: Use rclpy
# LOG_STD: Use standard output


class Profile(object):
    MODE_DISABLED = 0
    MODE_ROS2 = 1
    MODE_STD = 2
    LEVELS = ['DEBUG', 'STATS', 'INFO', 'WARN', 'ERROR']

    def __init__(self, mode, **kwargs):
        self.mode = mode
        # If None, then all sessions are enabled.
        self.enabled_sessions = None
        self.disabled_sessions = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'level'):
            self.level = 'INFO'
        assert(self.level in Profile.LEVELS)
        self._level_id = Profile.LEVELS.index(self.level)
        if self.mode == Profile.MODE_ROS2:
            self.node.get_logger().set_level(self._ros_level(self.level))

    @staticmethod
    def _ros_level(level):
        from rclpy.logging import LoggingSeverity
        if level == 'DEBUG':
            return LoggingSeverity.DEBUG
        elif level == 'INFO':
            return LoggingSeverity.INFO
        elif level == 'WARN':
            return LoggingSeverity.WARN
        elif level == 'ERROR':
            return LoggingSeverity.ERROR
        else:
            raise ValueError(f'Invalid logging level: {level}')

    def above_water_level(self, level):
        index = Profile.LEVELS.index(level)
        return index >= self._level_id


_g_config = Profile(Profile.MODE_STD)


def config(config, sessions=None):
    global _g_config
    _g_config = config
    enable_sessions(sessions)


def is_session_enabled(session):
    global _g_config
    return ((_g_config.enabled_sessions is None) or
            (session in _g_config.enabled_sessions))


def enable_sessions(sessions):
    global _g_config
    if sessions is None:
      _g_config.enabled_sessions = None
      _g_config.disabled_sessions.clear()
    else:
      if _g_config.enabled_sessions is None:
          _g_config.enabled_sessions = set()
      _g_config.enabled_sessions.update(sessions)
      _g_config.disabled_sessions -= set(sessions)

def disable_sessions(sessions):
    global _g_config
    if _g_config.enabled_sessions is not None:
      _g_config.enabled_sessions -= set(sessions)
    _g_config.disabled_sessions.update(sessions)

def info(arg1, arg2=None):
    message = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    return log('INFO', session, message)

def warn(arg1, arg2=None):
    message = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    return log('WARN', session, message)

def error(arg1, arg2=None):
    message = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    return log('ERROR', session, message)

def debug(arg1, arg2=None):
    message = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    return log('DEBUG', session, message)

def stats(arg1, arg2=None):
    message = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    return log('STATS', session, message)

def log(level, session, message):
    if _g_config.mode == Profile.MODE_DISABLED:
        return

    if not _g_config.above_water_level(level):
        return

    if not is_session_enabled(session):
        return

    if level == 'INFO':
        color_level = bcolors.OKGREEN
    elif level == 'WARN':
        color_level = bcolors.WARNING
    elif level == 'ERROR':
        color_level = bcolors.FAIL
    elif level == 'DEBUG':
        color_level = bcolors.OKCYAN + bcolors.UNDERLINE
    elif level == 'STATS':
        color_level = bcolors.OKBLUE
    else:
        color_level = bcolors.OKBLUE
    if _g_config.mode == Profile.MODE_STD:
        print(f'{color_level}[{session}]:\t{message}{bcolors.ENDC}')
    elif _g_config.mode == Profile.MODE_ROS2:
        line = f'{color_level}[{session}]:\t{message}{bcolors.ENDC}'
        if level == 'INFO':
            _g_config.node.get_logger().info(line)
        elif level == 'WARN':
            _g_config.node.get_logger().warn(line)
        elif level == 'ERROR':
            _g_config.node.get_logger().error(line)
        elif level == 'STATS':
            _g_config.node.get_logger().info(line)
        elif level == 'DEBUG':
            _g_config.node.get_logger().debug(line)
        else:
            return
    else:
        return
