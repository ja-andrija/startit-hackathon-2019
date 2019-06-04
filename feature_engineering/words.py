import csv
import pandas
import re
import util

def word_list(column, labels):
    words = [(l, re.findall(r'[^\s!,.?\'\]":;{}[0-9]+', s)) for (s, l) in zip(column, labels)]
    unique_words = [(l, list(set(ws))) for (l, ws) in words] 
    split_words = [[(w, l) for w in ws] for (l, ws) in unique_words]
    flat_words = [item for sublist in split_words for item in sublist]
    return flat_words

ImportantUpnpWords = [
    'DIAL',
    'microsoft',
    'Player',
    'TV ',
    'LG',
    'Philips',
    'Wireless',
    'bridge',
    'Lighting',
    'Camera',
    'Smart',
    'WebOSTV',
    'SystemProperties',
    'ZoneGroupTopology',
    'Streaming',
    'Amazon',
    'AlarmClock',
    'Google',
    'QPlay',
    'tencent-com',
    'sonos-com',
    'EmbeddedNetDeviceControl',
    'orange-com',
    'ScalarWebAPI',
    'NETGEAR',
    'Belkin',
    'remoteaccess',
     'BRAVIA',
    'VIErA',
    'KDL-',
    '(ReadyDLNA)',
    'RECEIVER',
    'QNAP',
    'BouygtelTV',
    'ZoneControl',
    'BboxTV',
    'EmbeddedNetDevice',
    'ManageableDevice',
    'ConfigurationManagement',
    'TurboNAS',
    'ReadyNAS',
    'RootNull',
    'cellvision',
    'Aficio',
    'Seagate',
    'Verizon',
    'Technology',
    'Miniserver',
    'PIONEER',
    'PacketVideo',
    'NVIDIA',
    'CANON',
    'Blu-ray',
    'Home',
    'Google Home',
    'Roku Streaming Player',
    'UTC',
    'MediaRenderer'
]

ImportantSsdpWords = [
    'IpBridge',
    'Windows',
    'TV',
    'Cling',
    'nanoleaf_aurora',
    'Mac',
    'LPUX',
    'Printer',
    'SecurityCamera',
    'schemas-wifialliance-org'
]

def create_word_columns(df, feature, important_words):
    column_names = list()
    # print(df[feature][0:5])
    # hack = [str(x) for x in df[feature]]
    for word in important_words:
        # print(word)
        column_name = feature + '_' + word
        column_names.append(column_name)
        df[column_name] = [str(x).find(word) >= 0 for x in df[feature]]

    return column_names

def create_ssdp_word_columns(df):
    return create_word_columns(df, 'ssdp', ImportantSsdpWords)


def create_upnp_word_columns(df):
    return create_word_columns(df, 'upnp', ImportantUpnpWords)
#df2 = util.load_data_to_dataframe('dataset/train.json')
#df2.drop(['ssdp'], 1)
#create_upnp_word_columns(df2)
#print(df2)
#df2.to_csv('hello.csv')


def test():
    df2 = util.load_data_to_dataframe('dataset/train.json')
    print(df2.count())
    # df2 = df2[df2['upnp'].isna()]
    # print("no upnp ", df2.count())
    dz = [str(x) for x in df2['dhcp']]
    dc = [str(x) for x in df2['device_class']]
    w = word_list(dz, dc)

    myFrame = pandas.DataFrame(w, columns = ['word', 'device_class'])
    newFrame = myFrame.groupby('word').agg("count")

    # print(newFrame.head())
    # print(newFrame[newFrame['device_class'] > 50])
#    print(newFrame[newFrame > 50].agg("count"))
    
    # print(newFrame)

    # print(myFrame.head())
    # print(len(w))
    
    with open('dhcp words.csv','w', newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['word','device_class'])
        for row in w:
            csv_out.writerow(row)

def create_columns(words_column):
    s = '[{"model_name": "Sonos Play:3", "model_description": "Sonos Play:3", "manufacturer": "Sonos, Inc.", "device_type": "urn:schemas-upnp-org:device:ZonePlayer:1", "services": ["urn:upnp-org:serviceId:AlarmClock", "urn:upnp-org:serviceId:MusicServices", "urn:upnp-org:serviceId:DeviceProperties", "urn:upnp-org:serviceId:SystemProperties", "urn:upnp-org:serviceId:ZoneGroupTopology", "urn:upnp-org:serviceId:GroupManagement", "urn:tencent-com:serviceId:QPlay"]}, {"model_name": "Sonos Play:3", "model_description": "Sonos Play:3 Media Server", "manufacturer": "Sonos, Inc.", "device_type": "urn:schemas-upnp-org:device:MediaServer:1", "services": ["urn:upnp-org:serviceId:ContentDirectory", "urn:upnp-org:serviceId:ConnectionManager"]}, {"model_name": "Sonos Play:3", "model_description": "Sonos Play:3 Media Renderer", "manufacturer": "Sonos, Inc.", "device_type": "urn:schemas-upnp-org:device:MediaRenderer:1", "services": ["urn:upnp-org:serviceId:RenderingControl", "urn:upnp-org:serviceId:ConnectionManager", "urn:upnp-org:serviceId:AVTransport", "urn:sonos-com:serviceId:Queue", "urn:upnp-org:serviceId:GroupRenderingControl"]}]'
    s2 = '[{"model_name": "LG Smart TV", "manufacturer": "LG Electronics", "device_type": "urn:schemas-upnp-org:device:Basic:1", "services": ["urn:lge-com:serviceId:webos-second-screen-3000-3001"]}, {"model_name": "LG Smart TV", "manufacturer": "LG Electronics", "device_type": "urn:dial-multiscreen-org:service:dial:1", "services": ["urn:dial-multiscreen-org:serviceId:dial"]}, {"model_name": "LG TV", "model_description": "LG WebOSTV DMRplus", "manufacturer": "LG Electronics.", "device_type": "urn:schemas-upnp-org:device:MediaRenderer:1", "services": ["urn:upnp-org:serviceId:AVTransport", "urn:upnp-org:serviceId:ConnectionManager", "urn:upnp-org:serviceId:RenderingControl"]}]'

    print(word_list([s, s2], ['TV', 'PC']))

# test()



