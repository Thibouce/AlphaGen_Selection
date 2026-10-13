"""Alpha Chunk Planner: calcule le découpage en chunks LTX d'une plaque pour le workflow décomposé de
Comfy Pro Alpha Generator.

Branchez les sorties aux widgets (ChunkPad start/length/end, Stitch plan, Plate/TrackMasks count, latent length...)
puis coupez (Ctrl+M) les branches de chunk dont l'index est >= `num_chunks`. Le texte `summary` dit lesquelles.
"""
from .chunking import MAX_CHUNKS, ltx_frames, plan_chunks  # noqa: F401
import json

SLOTS = 10  # nombre d'emplacements de chunk (sorties A0..A9 / B0..B9) du workflow


class AlphaChunkPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0, "max": 10**7, "tooltip": "Frame où le sujet est pointé (0 en général)"}),
                "chunk": ("INT", {"default": 49, "min": 9, "max": 145, "step": 8,
                                  "tooltip": "Longueur MAX d'un chunk (8n+1). 49 tient en 1920x1088 sur 32 Go."}),
                "overlap": ("INT", {"default": 7, "min": 0, "max": 72, "tooltip": "Recouvrement MIN entre chunks voisins"}),
                "mode": (["fixed", "shrink"], {"tooltip": "fixed: chunks de 'chunk' frames. shrink: raccourcit aussi les chunks quand le plan s'y prête"}),
                "render": (["full shot", "preview chunk"], {"tooltip": "preview chunk = un seul chunk à partir de 'start'"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "La plaque: donne le nombre de frames, la largeur et la hauteur"}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 10**7, "tooltip": "Si 'images' n'est pas branché"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    COMMON = [("count", "INT"), ("start", "INT"), ("last_frame", "INT"), ("length", "INT"), ("num_chunks", "INT"),
              ("end_a", "INT"), ("end_b", "INT"), ("width", "INT"), ("height", "INT"),
              ("latent_width", "INT"), ("latent_height", "INT"), ("plan", "STRING"), ("summary", "STRING")]
    OUTPUTS = (COMMON + [(f"A{i}", "INT") for i in range(SLOTS)] + [(f"B{i}", "INT") for i in range(SLOTS)]
               + [(f"on_{i}", "BOOLEAN") for i in range(SLOTS)])
    RETURN_NAMES = tuple(n for n, _ in OUTPUTS)
    RETURN_TYPES = tuple(t for _, t in OUTPUTS)
    FUNCTION = "plan"
    CATEGORY = "Comfy Compositing"
    DESCRIPTION = ("Chunk count / length / starts for the decomposed Comfy Pro Alpha workflow. "
                   "A* = ChunkPad starts for pass A (include 'start'), B* = pass B (isolated plate).")

    def plan(self, start, chunk, overlap, mode, render, images=None, frames=0, width=0, height=0):
        if images is not None:
            n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        else:
            n, h, w = int(frames), int(height), int(width)
        if n <= 0:
            raise ValueError("AlphaChunkPlanner: branchez 'images' ou renseignez 'frames'")
        if start >= n:
            raise ValueError(f"AlphaChunkPlanner: start ({start}) >= nombre de frames ({n})")
        count = n - start
        if render == "preview chunk":
            count = min(count, chunk)
        L, starts = plan_chunks(count, chunk, overlap, mode == "shrink")
        k = len(starts)
        if k > SLOTS:
            raise ValueError(f"AlphaChunkPlanner: {count} frames demandent {k} chunks mais le workflow n'a que {SLOTS} "
                             f"emplacements. Augmentez 'chunk', baissez 'overlap', ou ajoutez des branches.")
        pad = lambda xs: xs + [xs[-1]] * (SLOTS - len(xs))  # emplacements inutilisés: valeur valide, branche à couper
        a, b = pad([start + s for s in starts]), pad(starts)
        plan = json.dumps({"n": count, "h": h, "w": w, "chunk": L, "starts": starts})
        off = [i for i in range(SLOTS) if i >= k]
        summary = (f"{n} frames, depart {start}: {count} traitees -> {k} chunk(s) de {L} frames, starts {starts}, "
                   f"overlap min {min((L - (y - x) for x, y in zip(starts, starts[1:])), default=L)}. "
                   f"Actifs: chunks 0-{k - 1}" + (f" | a couper (Ctrl+M): chunks {off[0]}-{off[-1]}" if off else ""))
        print("[AlphaChunkPlanner]", summary)
        vals = (count, start, start + count - 1, L, k, start + count, count, w, h, w + (-w) % 32, h + (-h) % 32, plan, summary)
        on = tuple(i < k for i in range(SLOTS))  # True pour les chunks à garder actifs
        return {"ui": {"text": [summary]}, "result": vals + tuple(a) + tuple(b) + on}


class AlphaStitchAuto:
    """Remplace ComfyProAlpha_Stitch dans le workflow décomposé: reçoit les SLOTS sorties de chunk, mais
    n'évalue que les `num_chunks` premières (entrées « lazy »). Les branches des chunks suivants ne sont
    jamais exécutées: plus besoin de les couper à la main. Le node se développe ensuite en vrai
    ComfyProAlpha_Stitch relié aux seuls chunks utiles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"forceInput": True}),
                "num_chunks": ("INT", {"forceInput": True, "default": 1, "min": 1, "max": SLOTS}),
            },
            "optional": {f"chunk_{i}": ("*", {"lazy": True}) for i in range(SLOTS)},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "stitch"
    CATEGORY = "Comfy Compositing"

    @staticmethod
    def _linked(dynprompt, unique_id):
        ins = dynprompt.get_node(unique_id)["inputs"]
        return {k for k, v in ins.items() if isinstance(v, list) and len(v) == 2}

    def check_lazy_status(self, plan, num_chunks, dynprompt=None, unique_id=None, **kw):
        linked = self._linked(dynprompt, unique_id)
        return [f"chunk_{i}" for i in range(num_chunks) if f"chunk_{i}" in linked and kw.get(f"chunk_{i}") is None]

    def stitch(self, plan, num_chunks, dynprompt=None, unique_id=None, **kw):
        from comfy_execution.graph_utils import GraphBuilder  # importé ici: absent hors de ComfyUI
        inputs = dynprompt.get_node(unique_id)["inputs"]
        missing = [f"chunk_{i}" for i in range(num_chunks) if f"chunk_{i}" not in self._linked(dynprompt, unique_id)]
        if missing:
            raise ValueError(f"AlphaStitchAuto: {num_chunks} chunks prévus, entrées non branchées: {', '.join(missing)}")
        g = GraphBuilder()
        st = g.node("ComfyProAlpha_Stitch", plan=plan, **{f"chunk_{i}": inputs[f"chunk_{i}"] for i in range(num_chunks)})
        return {"result": (st.out(0),), "expand": g.finalize()}


NODE_CLASS_MAPPINGS = {"AlphaChunkPlanner": AlphaChunkPlanner, "AlphaStitchAuto": AlphaStitchAuto}
NODE_DISPLAY_NAME_MAPPINGS = {"AlphaChunkPlanner": "Alpha Chunk Planner", "AlphaStitchAuto": "Alpha Stitch Auto (lazy)"}
