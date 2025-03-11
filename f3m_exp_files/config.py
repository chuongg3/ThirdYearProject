import pathlib


_BASEDIR = pathlib.Path(__file__).parent.resolve()

# LLVM_DIR = _BASEDIR / 'llvm-project' / 'build' / 'bin'
# LLVM_DIR = '/home/chuongg3/Projects/TYP/build.clean/bin'
# LLVM_DIR = '/home/chuongg3/Projects/ThirdYearProject/build/bin'
LLVM_DIR = _BASEDIR / '..' / 'build' / 'bin'
LOGDIR = _BASEDIR / 'logs'
BMARK_DIR = _BASEDIR / 'benchmarks'

DBPATH = _BASEDIR / 'results.db'

BMARK_SUITES = [
        {'name': 'spec2006', 'pattern': ['4*.*']},
        {'name': 'spec2017', 'pattern': ['5*.*', '6*.*']},
        {'name': 'cc1plus', 'pattern': []},
        {'name': 'chrome', 'pattern': []},
        {'name': 'libreoffice', 'pattern': []},
        {'name': 'linux', 'pattern': []},
        {'name': 'llvm', 'pattern': []},]

TARGET = 'binary_dottext_size'
#TARGET = 'object_size'
