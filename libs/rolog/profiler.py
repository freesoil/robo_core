
import pstats
import time
import collections
from .session_log import is_session_enabled, stats
from .colors import bcolors
from .code_inspect import get_method_name

class Measure(object):

    def __init__(self):
        self.tick = time.time()

class DeltaMeasure(object):
    def __init__(self, prev_measure: Measure, next_measure: Measure):
        self.elapsed = next_measure.tick - prev_measure.tick

    def __str__(self):
        return f'{self.elapsed} s'

class Profiles(object):
    def __init__(self):
        self.sessions = collections.defaultdict(collections.OrderedDict)

    def __getitem__(self, key):
        return self.sessions[key]

_g_profiles = Profiles()
def stat(arg1, arg2=None):
    global _g_profiles
    name = arg1 if arg2 is None else arg2
    session = get_method_name(2) if arg2 is None else arg1
    if not isinstance(session, str):
        session = str(session)
    _g_profiles.sessions[session][name] = Measure()

def show_profile(session=None):
    session = get_method_name(2) if session is None else session
    if is_session_enabled(session):
        stats(session, cycle_profile(session))

def cycle_profile(session=None):
    global _g_profiles
    session = get_method_name(2) if session is None else session
    if not isinstance(session, str):
        session = str(session)
    start_measure = prev_measure = None
    NamedMeasure = collections.namedtuple('NamedMeasure', ['name', 'measure'])
    lines = []
    lines.append(f'=== Session [{session}] profile ===')
    for name, measure in _g_profiles.sessions[session].items():
        if prev_measure is not None:
            diff_measure = DeltaMeasure(prev_measure.measure, measure)
            lines.append(f'{prev_measure.name} -> {name}:\t{diff_measure}')
        else:
            start_measure = NamedMeasure(name=name, measure=measure)
        prev_measure = NamedMeasure(name=name, measure=measure)

    if start_measure is not None:
        diff_measure = DeltaMeasure(start_measure.measure, prev_measure.measure)
        lines.append(f'{bcolors.BOLD}Total:\t{diff_measure}{bcolors.ENDC}')

    del _g_profiles.sessions[session]

    return '\n'.join(lines)
