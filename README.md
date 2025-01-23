# CAMUS_Segmentation_Hybrid_Bottleneck

!pip install -q tensorflow==2.15.0

# Restart the runtime automatically to ensure TensorFlow is properly installed
import os
import IPython
os.kill(os.getpid(), 9)
