import numpy as np
def compute_recall_at_k(pred_pos, pred_neg, k):
    y_pred_pos, y_pred_neg = pred_pos.flatten().cpu().numpy(), pred_neg.flatten().cpu().numpy()
    num_pos = y_pred_pos.shape[0]
    num_neg_per_pos = y_pred_neg.shape[0] // num_pos
    successful_recalls = 0
    for i in range(num_pos):
        corresponding_neg_samples = y_pred_neg[i*num_neg_per_pos:(i+1)*num_neg_per_pos]
        combined_scores = np.concatenate(([y_pred_pos[i]], corresponding_neg_samples))
        sorted_indices = np.argsort(-combined_scores, axis=0)
        rank_of_positive = np.where(sorted_indices == 0)[0][0]
        if rank_of_positive < k:
            successful_recalls += 1
    recall_at_k = successful_recalls / num_pos
    return recall_at_k