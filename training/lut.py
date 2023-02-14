
import argparse
import json
import os
import sys
import utils
import torch 
sys.path.append("training/")
 
DB_DIR = "training/data"
NUM_CLASSES = 10
INPUT_DIMS = {"data": [1, 3, 32, 32]}
MIN_CONV_FEATURE_SIZE = 8
MIN_FC_FEATRE_SIZE    = 64
MEASURE_LATENCY_BATCH_SIZE = 128
MEASURE_LATENCY_SAMPLE_TIMES = 500
def parse_args(model):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=model,
        help="Multiple models for benchmarking (separate with comma)",
    )
    parser.add_argument("--db_dir", type=str, default=DB_DIR)
    parser.add_argument(
        "--input_dims",
        type=lambda x: json.loads(x),
        default=json.dumps(INPUT_DIMS),
    )
    parser.add_argument("--input_dtype", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="Jetson Nano",
    )
    args = parser.parse_args()
    return args


def get_net_runtime(model, build_lookup_table=True):
    lookup_table_path = "data/lut_{}".format(model)
    for layer_name, layer_properties in model.items():
        print(layer_name)
        print('    ', layer_properties, '\n')
    print('-------------------------------------------')
    
    num_w = utils.compute_resource(model, 'WEIGHTS')
    flops = utils.compute_resource(model, 'FLOPS')
    num_param = utils.compute_resource(model, 'WEIGHTS')

    model = model.cuda()
    
    print('Building latency lookup table for', 
          torch.cuda.get_device_name())
    if build_lookup_table:
        utils.build_latency_lookup_table(model, lookup_table_path=lookup_table_path, 
            min_fc_feature_size=MIN_FC_FEATRE_SIZE, 
            min_conv_feature_size=MIN_CONV_FEATURE_SIZE, 
            measure_latency_batch_size=MEASURE_LATENCY_BATCH_SIZE,
            measure_latency_sample_times=MEASURE_LATENCY_SAMPLE_TIMES,
            verbose=True)
    print('-------------------------------------------')
    print('Finish building latency lookup table.')
    print('    Device:', torch.cuda.get_device_name())
    print('    Model: ', model)    
    print('-------------------------------------------')
    
    latency = utils.compute_resource(model, 'LATENCY', lookup_table_path)
    print('Computed latency:     ', latency)
    latency = utils.measure_latency(model, 
        [MEASURE_LATENCY_BATCH_SIZE, INPUT_DIMS["data"]])

    return model

"""
op_time, net_time, all_found = get_net_runtime(
            model,
            blobs,
            args.input_dims,
            args.input_dtype,
            args.device,
            verbose=True,
        )
"""