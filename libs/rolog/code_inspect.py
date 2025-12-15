import inspect

def get_method_name(stack_pos):
    frame = inspect.currentframe()
    count = 0
    while frame:
        if count == stack_pos:
            break
        frame = frame.f_back
        count += 1

    local_vars = frame.f_locals
    method_name = frame.f_code.co_name
    if 'self' in local_vars:
        class_name = local_vars['self'].__class__.__name__
        method_name = f'{class_name}.{method_name}'
    return method_name
