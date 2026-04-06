# Known optima subpackage - provides benchmark reference values
from .macmpec import (
    KNOWN_OPTIMA,
    RED_FLAG_PROBLEMS,
    get_known_optimum,
    compute_optimality_gap,
)
from .nosbench import (
    DEFAULT_REFERENCE_FILENAME,
    load_reference,
    get_reference_path,
    get_known_optimum_nosbench,
    compute_optimality_gap_nosbench,
    set_reference_dict,
)
