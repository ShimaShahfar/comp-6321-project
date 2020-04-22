import os


def make_unique_path(path, seperator='-', leading_zeros=0, start_num=2):
    if not os.path.exists(path):
        return path
    path = os.path.abspath(path)
    base = path.split(os.path.sep)[-1].split('.')
    ext = ''
    if (len(base) > 1):
        ext = "." + base[-1]
        path = path[:-len(ext)-1]
    fspec = f"{path}{seperator}{{:0{leading_zeros+1}d}}" + ext
    for i in range(start_num, 100):
        npath = fspec.format(i)
        if not os.path.exists(npath):
            return npath
