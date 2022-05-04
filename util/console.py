import time


def msg_time(msg):
    print('\r' + msg + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def progress(current, total):
    print("\rprogress:" + format((current / total) * 100, '.2f') + "%，current" + str(
        current) + "，total：" + str(
        total), end="")
