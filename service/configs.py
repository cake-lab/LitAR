# To use a different quality setting, you need to comment out the current setting,
# and then uncomment the desired setting.

# System configurationsPORT = 8753

# Quality Setting: Low
MLP_SEQUENCE = [256, 128, 32]
PANORAMA_WIDTH = 512
PANORAMA_HEIGHT = 256

# Quality Setting: Medium
# MLP_SEQUENCE = [384, 192]
# PANORAMA_WIDTH = 768
# PANORAMA_HEIGHT = 384

# Quality Setting: High
# MLP_SEQUENCE = [512, 256, 128]
# PANORAMA_WIDTH = 1024
# PANORAMA_HEIGHT = 512

# Far field setting
N_ANCHORS = 1280
N_ANCHOR_NEIGHBORS = 32

CUBEMAP_SIZE = 512

# NEAR_FIELD_SAMPLE_RATE = 0.005 # 5 millimeter
NEAR_FIELD_SIZE = 20
NEAR_FIELD_SIZE_HALF = NEAR_FIELD_SIZE / 2
NEAR_FIELD_CLIP_DST = 1000
