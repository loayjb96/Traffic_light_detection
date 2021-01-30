import datetime


def timer_decorator(func):
    def inner(*args):
        before = datetime.datetime.now()
        res = func(*args)
        after = datetime.datetime.now()
        print("args" , args)
        print(f"{func.__name__} for frame {args[0].curr_container.frame_id} runtime: {after - before}")
        return res
    return inner
