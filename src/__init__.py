import os
import sys

# _TEST_ROOT = os.path.dirname(__file__)  # root of test folder
# _PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
# _PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data

# _CURRENT_ROOT = os.path.abspath(os.path.dirname(__file__))
# _PROJECT_ROOT = os.path.dirname(_CURRENT_ROOT)
# _PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data
# _TEST_ROOT = os.path.join(_PROJECT_ROOT, "tests")  # root of data

# sys.path.append(_TEST_ROOT)
# from tests.test_data import test_data



_CURRENT_ROOT = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_ROOT)
_TEST_ROOT = os.path.join(_PROJECT_ROOT, "tests")
sys.path.append(_TEST_ROOT)

from tests.test_data import test_data