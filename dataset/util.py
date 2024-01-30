from sklearn.preprocessing import MinMaxScaler, StandardScaler
from absl import logging

def normalize(x, feat_norm_type='min_max'):
    if feat_norm_type == 'min_max':
        scaler = MinMaxScaler([0, 1])
    elif feat_norm_type == 'standard':
        scaler = StandardScaler()
    else:
        logging.fatal('Unknown normalization type.') 
    norm_x = scaler.fit_transform(x)
    return norm_x