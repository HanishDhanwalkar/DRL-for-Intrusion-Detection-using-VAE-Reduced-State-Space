import pandas as pd

train_file = "data/KDDTrain+.txt"
test_file = "data/KDDTest+.txt"

column_names = [
            'duration',
            'protocol_type',
            'service',
            'flag', 
            'src_bytes', 
            'dst_bytes',
            'land', 
            'wrong_fragment', 
            'urgent', 
            'hot', 
            'num_failed_logins',
            'logged_in', 
            'num_compromised', 
            'root_shell', 
            'su_attempted', 
            'num_root',
            'num_file_creations', 
            'num_shells', 
            'num_access_files', 
            'num_outbound_cmds',
            'is_host_login', 
            'is_guest_login', 
            'count', 
            'srv_count', 
            'serror_rate',
            'srv_serror_rate', 
            'rerror_rate', 
            'srv_rerror_rate', 
            'same_srv_rate',
            'diff_srv_rate', 
            'srv_diff_host_rate', 
            'dst_host_count', 
            'dst_host_srv_count',
            'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 
            'dst_host_serror_rate', 
            'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', 
            'class', #or some paper calls this - attack_type 
            'difficulty_level'
            ]

def read_data(filepath, col, is_feature):
    df = pd.read_csv(filepath, names=col)
    df2 = df.dropna()
    if is_feature:
        df2 = df2.drop(['service', 'flag'], axis=1)
        df2 = df2.drop(['class'], axis=1) # Label
        df2['protocol_type'] = df2['protocol_type'].map({'tcp': 1, 'udp': 2, 'icmp': 3})

    else:
        df2 = df2[['class']]
        df2['class'] = df2['class'].map({'normal': 0}).fillna(1).astype(int)
    return df2

def train_data(filepath = train_file, col = column_names):
    TrainX = read_data(filepath, col, True)
    TrainY = read_data(filepath, col, False)
    return TrainX.to_numpy() , TrainY.to_numpy()

def test_data(filepath = test_file, col = column_names):
    TestX = read_data(filepath, col, True)
    TestY = read_data(filepath, col, False)

    return TestX.to_numpy() , TestY.to_numpy()

# X ,y = train_data()
# print(X)
# print(y)