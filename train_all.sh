# !/bin/bash

# python3 train.py --config ./configs/dorsal_stream_r3d18.yaml
# python3 test.py --config ./configs/dorsal_stream_r3d18.yaml

# python3 train.py --config ./configs/dorsal_stream_dorsalnet.yaml
# python3 test.py --config ./configs/dorsal_stream_dorsalnet.yaml

# python3 train.py --config ./configs/dorsal_stream_gabor3d.yaml
# python3 test.py --config ./configs/dorsal_stream_gabor3d.yaml

# python3 train.py --config ./configs/dorsal_stream_resnet18.yaml
# python3 test.py --config ./configs/dorsal_stream_resnet18.yaml

python3 train.py --config ./configs/dorsal_stream_simple3d1.yaml
python3 test.py --config ./configs/dorsal_stream_simple3d1.yaml

python3 train.py --config ./configs/dorsal_stream_simple3d3.yaml
python3 test.py --config ./configs/dorsal_stream_simple3d3.yaml

# python3 train.py --config ./configs/dorsal_stream_simple3d5.yaml
# python3 test.py --config ./configs/dorsal_stream_simple3d5.yaml

python3 train.py --config ./configs/dorsal_stream_simple3d7.yaml
python3 test.py --config ./configs/dorsal_stream_simple3d7.yaml

