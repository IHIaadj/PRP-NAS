'''Base Supernetwork Class'''
import torch
import torch.nn as nn 

from proxylessnas.proxylessnas.model_zoo import proxyless_base

types = ["201", "DARTS", "ProxylessNAS"]
class Supernetwork(nn.Module):

    def _base(self, pretrained=True, net_config=None, net_weight=None):
        assert net_config is not None, "Please input a network config"
        net_config_path = download_url(net_config)
        net_config_json = json.load(open(net_config_path, 'r'))
        if net_config_json['name'] == ProxylessNASNets.__name__:
            net = ProxylessNASNets.build_from_config(net_config_json)
        if 'bn' in net_config_json:
            net.set_bn_param(
                bn_momentum=net_config_json['bn']['momentum'],
                bn_eps=net_config_json['bn']['eps'])

        if pretrained:
            assert net_weight is not None, "Please specify network weights"
            init_path = download_url(net_weight)
            init = torch.load(init_path, map_location='cpu')
            net.load_state_dict(init['state_dict'])

        return net