"""Calcul automatique du nombre, de la durée et des débuts des chunks LTX selon la longueur de la plaque.

Contraintes (d'après le node ComfyProAlphaGenerator): longueur de chunk = 8k+1 (9..145), au plus 24 chunks,
les chunks voisins se recouvrent de `overlap` frames et sont fondus au raccord.
ATTENTION: ce n'est pas le _engine.chunk_starts d'origine (non fourni), mais un calcul équivalent en intention.
"""
import math

MAX_CHUNKS = 24


def ltx_frames(n):
    """Plus petit 8k+1 >= n (longueur LTX valide), minimum 1."""
    return 8 * math.ceil((max(n, 1) - 1) / 8) + 1


def _even_starts(count, L, K):
    return [round(i * (count - L) / (K - 1)) for i in range(K)]


def _min_overlap(starts, L):
    return min((L - (b - a) for a, b in zip(starts, starts[1:])), default=L)


def plan_chunks(count, lmax=49, omin=7, shrink=False):
    """-> (L, starts): longueur de chunk et débuts (dans le repère de la plage traitée).

    - count <= lmax : un seul chunk, de longueur 8k+1 >= count (la fin est complétée par le node).
    - sinon: K = nombre minimal de chunks de longueur lmax qui couvrent `count` avec un recouvrement >= omin,
      débuts répartis régulièrement (le dernier chunk finit exactement sur la dernière frame).
    - shrink=True: raccourcit aussi L (au plus petit 8k+1 suffisant pour K chunks): moins de frames calculées,
      mais s'écarte des 49 frames pour lesquelles les réglages d'origine sont calés (non testé sur la qualité).
    """
    if lmax < 9 or (lmax - 1) % 8:
        raise ValueError("chunk (lmax) doit valoir 8k+1 et être >= 9")
    if not 0 <= omin < lmax:
        raise ValueError("overlap doit être compris entre 0 et chunk-1")
    if count <= lmax:
        return min(lmax, ltx_frames(count)), [0]
    K = math.ceil((count - omin) / (lmax - omin))
    while True:
        L = lmax
        if shrink:
            L = min(lmax, max(9, ltx_frames(math.ceil((count + (K - 1) * omin) / K))))
        starts = _even_starts(count, L, K)
        while _min_overlap(starts, L) < omin and L < lmax:  # l'arrondi a réduit un recouvrement
            L += 8
            starts = _even_starts(count, L, K)
        if _min_overlap(starts, L) >= omin:
            break
        K += 1
    if K > MAX_CHUNKS:
        raise ValueError(f"{count} frames demandent {K} chunks (max {MAX_CHUNKS}): augmenter chunk, baisser overlap ou couper le plan")
    return L, starts


def legacy_chunk_starts(count, L, step):
    """Ancien calcul (pas fixe, dernier chunk ramené sur la fin)."""
    starts = [0]
    while starts[-1] + L < count:
        starts.append(min(starts[-1] + step, count - L))
    return starts


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    for shrink in (False, True):
        L, s = plan_chunks(n, shrink=shrink)
        print(f"{'shrink' if shrink else 'fixed '} L={L} K={len(s)} starts={s} overlap_min={_min_overlap(s, L)}")
