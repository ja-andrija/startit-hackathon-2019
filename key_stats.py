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
    
    

def get_dhcp_stats(json_path, tag_name = "dhcp"):
    value1_name ="paramlist"
    value2_name = "classid"
    
    data = read_proper_json_from_file(json_path)

    # Device name -> paramlist, classId, stat_device_tag["CLASS"]["paramList"][xyz]" = number
    # Device name -> paramlist, classId, stat_device_tag["CLASS"]["classid"][xyz]" = number
    stat_device_tag = dict()

    # dict["paramlist"]["xyz"]"["CLASS"] = number
    stat_tag_device = dict()
    stat_tag_device[value1_name] = dict()
    stat_tag_device[value2_name] = dict()

    for dp in data:
        dev_class = dp["device_class"]

        if tag_name in dp.keys():
            if value1_name in dp[tag_name][0].keys():
                if dp[tag_name][0][value1_name] not in stat_tag_device[value1_name].keys():
                    stat_tag_device[value1_name][dp[tag_name][0][value1_name]] = dict()

                if dev_class not in stat_tag_device[value1_name][dp[tag_name][0][value1_name]].keys():
                    stat_tag_device[value1_name][dp[tag_name][0][value1_name]][dev_class] = 0

                stat_tag_device[value1_name][dp[tag_name][0][value1_name]][dev_class] += 1

                if dev_class not in stat_device_tag.keys():
                    stat_device_tag[dev_class] = dict()

                    stat_device_tag[dev_class][value1_name] = dict()
                    stat_device_tag[dev_class][value2_name] = dict()

                if dp[tag_name][0][value1_name] not in stat_device_tag[dev_class][value1_name].keys():
                    stat_device_tag[dev_class][value1_name][dp[tag_name][0][value1_name]] = 0

                stat_device_tag[dev_class][value1_name][dp[tag_name][0][value1_name]] += 1

            if value2_name in dp[tag_name][0].keys():
                if dp[tag_name][0][value2_name] not in stat_tag_device[value2_name].keys():
                    stat_tag_device[value2_name][dp[tag_name][0][value2_name]] = dict()

                if dev_class not in stat_tag_device[value2_name][dp[tag_name][0][value2_name]].keys():
                    stat_tag_device[value2_name][dp[tag_name][0][value2_name]][dev_class] = 0

                stat_tag_device[value2_name][dp[tag_name][0][value2_name]][dev_class] += 1

                if dp[tag_name][0][value2_name] not in stat_device_tag[dev_class][value2_name].keys():
                    stat_device_tag[dev_class][value2_name][dp[tag_name][0][value2_name]] = 0

                stat_device_tag[dev_class][value2_name][dp[tag_name][0][value2_name]] += 1

    for device_class, field_types in stat_device_tag.items():
        for field_type, values in field_types.items():
                field_types[field_type] = dict(sorted(values.items(), key = itemgetter(1), reverse=True))
        stat_device_tag[device_class] = field_types

    write_json_to_file(stat_tag_device, r'C:\Users\vr1\startit-hackathon-2019\key_stats_dhcp_tags_dev.json')
    write_json_to_file(stat_device_tag, r'C:\Users\vr1\startit-hackathon-2019\key_stats_dhcp_dev_tags.json')
    return stat_device_tag