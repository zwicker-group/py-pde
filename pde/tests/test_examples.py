'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import glob
import sys
import os
import subprocess as sp
from pathlib import Path
from typing import List  # @UnusedImport

import pytest
import numba as nb

from ..tools.misc import module_available, skipUnlessModule



PACKAGE_PATH = Path(__file__).resolve().parents[2]
EXAMPLES = glob.glob(str(PACKAGE_PATH / 'examples' / '*.py'))
NOTEBOOKS = glob.glob(str(PACKAGE_PATH / 'examples' / 'jupyter' / '*.ipynb'))

SKIP_EXAMPLES: List[str] = ['make_movie.py']
if not module_available("matplotlib"):
    SKIP_EXAMPLES.append('trackers.py')


@pytest.mark.skipif(sys.platform == 'win32', reason="Assumes unix setup")
@pytest.mark.skipif(nb.config.DISABLE_JIT,
                    reason='pytest seems to check code coverage')
@pytest.mark.parametrize('path', EXAMPLES)
def test_example(path):
    """ runs an example script given by path """
    if os.path.basename(path).startswith('_'):
        pytest.skip('skip examples starting with an underscore')
    if any(name in path for name in SKIP_EXAMPLES):
        pytest.skip(f'Skip test {path}')
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE,
                    stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()            
        
    msg = 'Script `%s` failed with following output:' % path
    if outs:
        msg = '%s\nSTDOUT:\n%s' % (msg, outs)
    if errs:
        msg = '%s\nSTDERR:\n%s' % (msg, errs)
    assert proc.returncode == 0, msg



@skipUnlessModule('jupyter')
@pytest.mark.skipif(nb.config.DISABLE_JIT,
                    reason='pytest seems to check code coverage')
@pytest.mark.parametrize('path', NOTEBOOKS)
def test_jupyter_notebooks(path, tmp_path):
    """ run the jupyter notebooks """
    if os.path.basename(path).startswith('_'):
        pytest.skip('skip examples starting with an underscore')
        
    # adjust python environment
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + my_env["PATH"]        
        
    outfile = tmp_path / os.path.basename(path)
    sp.check_call([sys.executable, '-m', 'jupyter', 'nbconvert', 
                   '--to', 'notebook', '--output', outfile,
                   '--execute', path], env=my_env)
