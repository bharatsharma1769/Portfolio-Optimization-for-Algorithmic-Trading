```python
%load_ext autoreload
%autoreload 2

import sys    
from pathlib import Path    
# make src importable
sys.path.append(str(Path().resolve().parents[0] / "src"))

from common_imports import *
from bootstrap import bootstrap
from utils import *


# initialize project
ROOT, DATA_DIR, FIGS_DIR, MODELS_DIR, LOGS_DIR = bootstrap(seed=1)
```


```python
print("âœ… Setup complete")
```

    âœ… Setup complete
    
