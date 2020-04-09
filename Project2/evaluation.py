def evaluate(TP, FP, FN, correction=True):
    if correction == True:
        precision = (TP + 1) / (TP + FP + 1)  # Laplace Correction is applied
        recall = (TP + 1) / (TP + FN + 1)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def compute_accuracy(y_hat, y_gold):
    cnt = 0
    N = len(y_hat)
    for i in range(N):
        if y_hat[i] == y_gold[i]:
            cnt += 1
    return cnt / N