from util import *


def get_key_stats(json_path):
    data = read_proper_json_from_file(json_path)
    stats = {}
    macs = set()
    for dp in data:
        for key in dp.keys():
            if key not in stats:
                stats[key] = 0
            stats[key] += 1
            if key == 'mac':
                macs.add(dp[key])
    return stats, macs