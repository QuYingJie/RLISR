import numpy as np

def cal_precision(preds):
    """
    calculate average precision (area under PR curve)
    preds: list, element is binary (0 / 1)
    """
    l = len(preds)
    sum = 0.0
    for i in range(l):
        sum += (np.sum(preds[i]) / len(preds[i]))
    return sum / l

def cal_recall(preds, trues):
    l = len(preds)
    sum = 0.0
    for i in range(l):
        sum += (np.sum(preds[i]) / trues[i])
    return sum / l

def cal_ndcg(preds):
    l = len(preds)
    sum = 0.0
    for i in range(l):
        pred = preds[i]
        dcg = 0.0
        idcg = np.sum(1.0 / np.log2(np.arange(2, np.sum(pred) + 2)))
        for j in range(len(pred)):
            if pred[j] == 1:
                dcg += 1.0 / np.log2(j + 2)
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg
        sum += ndcg
    return sum / l

def cal_f1(preds, trues):
    l = len(preds)
    sum = 0.0
    for i in range(l):
        pred = preds[i]
        true = trues[i]
        p = np.sum(pred) / len(pred)
        r = np.sum(pred) / true
        if (p + r) == 0.0:
            f1 = 0.0
        else:
            f1 = (2.0 * p * r) / (p + r)
        sum += f1
    return sum / l

def cal_map(preds, trues):
    l = len(preds)
    sum = 0.0
    for i in range(l):
        pred = preds[i]
        true = trues[i]
        t = min(len(pred), true)
        pk = []
        for j in range(len(pred)):
            pk.append(np.sum(pred[:j + 1]))
        s = 0.0
        for i in range(len(pk)):
            s += ((pk[i] / (i + 1)) * pred[i])
        sum += (s / t)
    return sum / l

if __name__ == '__main__':
    preds = [2,4]
    trues = [3,7]
    r = cal_recall(preds, trues)
    print(r)