def ap_at_k(preds, positive, k=300, return_wa=False):
    """
    Compute average precision at k.
    Args:
        preds: (Sequence) IDs of predictions sorted in descending order of positive likelihood scores.
        positive: (Sequence) IDs of positive samples.
    """
    preds = preds[:min(len(preds), k)]
    scores = 0.
    wa = []
    TP = 0
    
    for count, pred_id in enumerate(preds):
        if pred_id in positive:
            TP += 1
            scores += TP/(count + 1)
        else:
            wa += [pred_id]
    if return_wa:
        return scores, wa
    else:
        return scores