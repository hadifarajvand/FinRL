#!/usr/bin/env python3
"""
Test script to verify FinRL works in the drl_env_new kernel
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import finrl
    print(f"✅ FinRL imported successfully")
    # Try to get version, but don't fail if it doesn't exist
    try:
        print(f"   Version: {finrl.__version__}")
    except AttributeError:
        print(f"   Version: Not available")
except ImportError as e:
    print(f"❌ Failed to import FinRL: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully: {torch.__version__}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")

try:
    import gymnasium
    print(f"✅ Gymnasium imported successfully: {gymnasium.__version__}")
except ImportError as e:
    print(f"❌ Failed to import Gymnasium: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas imported successfully: {pd.__version__}")
except ImportError as e:
    print(f"❌ Failed to import Pandas: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully: {np.__version__}")
except ImportError as e:
    print(f"❌ Failed to import NumPy: {e}")

print("\n🎉 Kernel test completed!") 