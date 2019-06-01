from util import *

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