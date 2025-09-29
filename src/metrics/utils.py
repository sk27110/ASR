import editdistance


def cer(target: str, pred: str) -> float:
    if not target:
        return 1.0
    return editdistance.eval(target, pred) / len(target)


def wer(target: str, pred: str) -> float:
    t_words = target.split()
    p_words = pred.split()
    if len(t_words) == 0:
        return 1.0
    return editdistance.eval(t_words, p_words) / len(t_words)
