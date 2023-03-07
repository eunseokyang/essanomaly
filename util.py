import numpy as np
import torch

# V, I, SOC, T, Vgap
# Min-max scale with (0.01, 0.99) quantile
PANLI_NORMALIZER = np.array([[803., -337., 0., 25., 0.], 
                             [972., 510., 90., 38., 0.03]])

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def point_adjustment(gt, pr, anomaly_rate=0.05):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)])
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(int)

    # pa
    for s, e in intervals:
        interval = slice(s, e)
        if pa[interval].sum() > 0:
            pa[interval] = 1

    # confusion matrix
    TP = (gt * pa).sum()
    TN = ((1 - gt) * (1 - pa)).sum()
    FP = ((1 - gt) * pa).sum()
    FN = (gt * (1 - pa)).sum()

    assert (TP + TN + FP + FN) == len(gt)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*precision*recall/(precision+recall)
    
    print(f"anomaly rate: {anomaly_rate:.3f} | precision: {precision:.5f} | recall: {recall:.5f} | F1-score: {f1_score:.5f}")

    return precision, recall, f1_score, pa


def evaluate_with_pa(gt, pred, rates):
    n_tops = (rates * len(gt)).astype(int)
    anomaly_args = np.argwhere(gt).flatten()
    
    anomaly = []
    sorted_values = np.sort(pred)[::-1]
    
    for n_top, rate in zip(n_tops, rates):
        thres = sorted_values[n_top]
        anomaly_pts = np.argwhere(pred > thres).flatten()
        terms = anomaly_pts[1:] - anomaly_pts[:-1]
        terms = terms > 1
        
        sequence_args = np.argwhere(terms).flatten() + 1
        sequence_length = list(sequence_args[1:] - sequence_args[:-1])
        sequence_args = list(sequence_args)
        
        sequence_args.insert(0, 0)
        if len(sequence_args) > 1:
            sequence_length.insert(0, sequence_args[1])
        sequence_length.append(len(anomaly_args) - sequence_args[-1])
        
        sequence_args = anomaly_pts[sequence_args]
        _sequence_args = sequence_args + np.array(sequence_length)
        
        anomaly.append(np.array((sequence_args, _sequence_args)))
        
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    
    print('# anomalies :', len(anomaly_label_seq))
    for _seq in anomaly_label_seq:
        print(_seq, ', length : ', _seq[1]-_seq[0], sep='')
    print()
    
    precision = []
    recall = []
    
    for seq in anomaly:
        overlap_table = (seq[[0]] < anomaly_label_seq[:, [1]]) & (seq[[1]] > anomaly_label_seq[:, [0]])
        overlap_table = overlap_table.astype(int)

        _precision = overlap_table.sum(axis=0) > 0
        precision.append(len(_precision[_precision]) / len(_precision))

        _recall = overlap_table.sum(axis=1) > 0
        recall.append(len(_recall[_recall]) / len(_recall))
        
    for rate, prec, rec in zip(rates, precision, recall):
        print('threshold : ', rate*100, '%, precision : ', prec, ', recall : ', rec, sep='')
        
    f1_score = [2*p*r/(p+r) for p,r in zip(precision, recall)]
        
    return precision, recall, f1_score