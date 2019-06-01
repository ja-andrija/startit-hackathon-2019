from util import *
from collections import defaultdict
from operator import itemgetter

def get_key_stats(json_path):
    data = read_proper_json_from_file(json_path)
    stats = {}
    unique_macs = set()
    for dp in data:
        for key in dp.keys():
            if key not in stats:
                stats[key] = 0
            stats[key] += 1
            if key == 'mac':
                unique_macs.add(dp[key])
    return stats, unique_macs

def get_octet_mac_stats(json_path):
    data = read_proper_json_from_file(json_path)
    unique_octets = set()
    for dp in data:
        mac = dp['mac']
        mac_octets = mac.split(':')
        unique_octets.add(':'.join(mac_octets[:3]))
    return unique_octets

def get_octet_class_stats(json_path):
    data = read_proper_json_from_file(json_path)
    mac_octets = {}
    for dp in data:
        mac = dp['mac']
        mac_octet = mac.split(':')
        mac_octet = ":".join(mac_octet[:3])
        if mac_octet not in mac_octets:
            mac_octets[mac_octet] = {}
        dev_class = dp['device_class']
        if dev_class not in mac_octets[mac_octet]:
            mac_octets[mac_octet][dev_class] = 0
        mac_octets[mac_octet][dev_class] +=1
    return mac_octets

def get_mdns_stats(json_path):
    data = read_proper_json_from_file(json_path)
    stats_dev_count = {}
    stats_dev_to_mdns = {}
    stats_mdns_to_dev = {}
    for dp in data:
        dev_class = dp["device_class"]
        if dev_class not in stats_dev_count.keys():
            stats_dev_count[dev_class] = {
                "pres": 0,
                "not": 0
            }
        if dev_class not in stats_dev_to_mdns.keys():
            stats_dev_to_mdns[dev_class] = {}
        if "mdns_services" in dp.keys():
            stats_dev_count[dev_class]["pres"] += 1
            for key in dp["mdns_services"]:
                if key not in stats_mdns_to_dev.keys():
                    stats_mdns_to_dev[key] = {}
                if dev_class not in stats_mdns_to_dev[key]:
                    stats_mdns_to_dev[key][dev_class] = 0
                if key not in stats_dev_to_mdns[dev_class]:
                    stats_dev_to_mdns[dev_class][key] = 0
                stats_mdns_to_dev[key][dev_class] += 1
                stats_dev_to_mdns[dev_class][key] +=1
        else:
            stats_dev_count[dev_class]["not"] += 1
        
        stats_dev_to_mdns_out = {}
        for (key, val) in stats_dev_to_mdns.items():
            stats_dev_to_mdns_out[key] = sorted(val.items(), key=itemgetter(1), reverse=True)
        stats_mdns_to_dev_out = {}
        for key, val in stats_mdns_to_dev.items():
            stats_mdns_to_dev_out[key] = sorted(val.items(), key=itemgetter(1), reverse=True)
        
    return stats_dev_count, stats_dev_to_mdns_out, stats_mdns_to_dev_out
    