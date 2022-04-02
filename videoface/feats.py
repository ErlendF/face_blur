def remove_feats(seqs):
    for i in range(len(seqs)):
        seqs[i] = [face['bbox'] for face in seqs[i]]
    return seqs
