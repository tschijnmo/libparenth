import os.path
def FlagsForFile(filename, **kwargs):
    proj_dir = os.path.dirname(os.path.realpath(__file__))

    proj_include = os.path.join(proj_dir, 'include');
    fbitset_include = os.path.join(proj_dir, 'deps', 'fbitset', 'include')
    flags = [
            '-std=gnu++1z', '-I/usr/local/include',
            '-I' + proj_include, '-I' + fbitset_include,
            '-I.'
    ]
    return {'flags': flags}
