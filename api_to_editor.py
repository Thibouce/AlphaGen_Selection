#!/usr/bin/env python3
"""Convertit un workflow API (build_workflow.py) en workflow format éditeur, node par node, pour l'ouvrir
et le modifier dans ComfyUI. Il lit les définitions de nodes de VOTRE ComfyUI (ordre des widgets, sorties).

  python api_to_editor.py workflow_api.json -o workflow_editor_full.json --url http://127.0.0.1:8188
  # ou, sans serveur joignable: ouvrir http://127.0.0.1:8188/object_info, enregistrer, puis
  python api_to_editor.py workflow_api.json --object-info object_info.json
"""
import argparse
import json
import urllib.request

PRIMS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
COL_W, GAP_X, GAP_Y = 380, 80, 30


def spec(info, name):
    for sect in ("required", "optional"):
        if name in info.get("input", {}).get(sect, {}):
            return info["input"][sect][name]
    return None


def input_order(info):
    return [(n, v) for sect in ("required", "optional") for n, v in info.get("input", {}).get(sect, {}).items()]


def kind(v):
    """-> (type, options, config, is_widget)"""
    t, cfg = v[0], (v[1] if len(v) > 1 and isinstance(v[1], dict) else {})
    if isinstance(t, list):
        return "COMBO", t, cfg, True
    if t == "COMBO":
        return "COMBO", cfg.get("options", []), cfg, True
    return t, None, cfg, t in PRIMS


def default(t, opts, cfg):
    if "default" in cfg:
        return cfg["default"]
    return {"COMBO": (opts or [""])[0] if opts else "", "BOOLEAN": False, "STRING": "", "INT": 0, "FLOAT": 0.0}.get(t)


T = lambda *t: [(x.lower(), x) for x in t]
KNOWN_OUT = {
    "LoadVideo": [("VIDEO", "VIDEO")], "GetVideoComponents": [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")],
    "UNETLoader": [("MODEL", "MODEL")], "UnetLoaderGGUF": [("MODEL", "MODEL")],
    "LTXICLoRALoaderModelOnly": [("model", "MODEL"), ("latent_downscale_factor", "FLOAT")],
    "ModelAttentionBackend": [("MODEL", "MODEL")], "CLIPLoader": [("CLIP", "CLIP")],
    "CLIPTextEncode": [("CONDITIONING", "CONDITIONING")], "VAELoader": [("VAE", "VAE")],
    "LTXVConditioning": [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
    "EmptyLTXVLatentVideo": [("LATENT", "LATENT")], "LTXVEmptyLatentAudio": [("Latent", "LATENT")],
    "RandomNoise": [("NOISE", "NOISE")], "KSamplerSelect": [("SAMPLER", "SAMPLER")], "ManualSigmas": [("SIGMAS", "SIGMAS")],
    "LTXAddVideoICLoRAGuide": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
    "LTXVConcatAVLatent": [("latent", "LATENT")], "CFGGuider": [("GUIDER", "GUIDER")],
    "SamplerCustomAdvanced": [("output", "LATENT"), ("denoised_output", "LATENT")],
    "LTXVSeparateAVLatent": [("video_latent", "LATENT"), ("audio_latent", "LATENT")],
    "LTXVCropGuides": [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
    "PreviewImage": [],
    # nodes du pack: nombre de sorties connu par l'usage dans le node
    "ComfyProAlpha_Refine": [("selection", "*"), ("refined", "*"), ("others", "*")],
    "ComfyProAlpha_Final": [("alpha", "MASK"), ("comp", "IMAGE")],
}


try:  # le node planificateur: sorties connues (nom, type)
    from comfyui_chunk_planner import AlphaChunkPlanner as _P
    KNOWN_OUT["AlphaChunkPlanner"] = list(zip(_P.RETURN_NAMES, _P.RETURN_TYPES))
except ImportError:
    pass

# Ordres d'inputs réels, relevés dans le code / les captures du node (le mode hors-ligne devine les autres).
KNOWN_INPUT_ORDER = {
    "ComfyProAlpha_Isolate": ["plate", "selection", "others", "grow", "blur", "start"],
    "ComfyProAlpha_ChunkPad": ["images", "start", "length", "end"],
    "ComfyProAlpha_Final": ["plate", "ltx", "selection", "refined", "others", "ltx_isolated", "near_px",
                            "filename_prefix", "format", "first_frame", "save", "qc", "comp"],
}


def offline_object_info(api):
    """Définitions déduites du workflow API lui-même (ordre des inputs = ordre d'appel dans le node).
    Moins fiable qu'/object_info: l'ordre des widgets des nodes SAM3 / ComfyProAlpha_* est supposé."""
    oi = {}
    for n in api.values():
        e = oi.setdefault(n["class_type"], {"input": {"required": {}}, "used": 0})
        for k, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in api:
                src = api[str(v[0])]["class_type"]
                ko = KNOWN_OUT.get(src, [])
                e["input"]["required"][k] = [ko[v[1]][1] if v[1] < len(ko) else "*"]
                oi.setdefault(src, {"input": {"required": {}}, "used": 0})["used"] = max(oi[src]["used"], v[1] + 1)
            elif isinstance(v, bool):
                e["input"]["required"][k] = ["BOOLEAN", {}]
            elif isinstance(v, int):
                e["input"]["required"][k] = ["INT", {"control_after_generate": k == "noise_seed"}]
            elif isinstance(v, float):
                e["input"]["required"][k] = ["FLOAT", {}]
            else:
                e["input"]["required"][k] = ["STRING", {"video_upload": True} if n["class_type"] == "LoadVideo" else {}]
    for ct, e in oi.items():
        order = KNOWN_INPUT_ORDER.get(ct)
        if order:
            req = e["input"]["required"]
            e["input"]["required"] = {k: req[k] for k in order if k in req} | {k: v for k, v in req.items() if k not in order}
        outs = KNOWN_OUT.get(ct) or [(f"out{i}", "*") for i in range(e.pop("used", 0))]
        e.pop("used", None)
        e["output_name"], e["output"] = [o[0] for o in outs], [o[1] for o in outs]
    return oi


def convert(api, oi):
    missing = sorted({n["class_type"] for n in api.values() if n["class_type"] not in oi})
    if missing:
        raise SystemExit("Nodes absents de ce ComfyUI (installer les packs puis relancer): " + ", ".join(missing))
    # profondeur topologique
    depth = {}

    def d(nid):
        if nid not in depth:
            ups = [v[0] for v in api[nid]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in api]
            depth[nid] = 1 + max((d(u) for u in ups), default=-1)
        return depth[nid]

    for nid in api:
        d(nid)
    order = sorted(api, key=lambda i: (depth[i], int(i)))
    colY = {}
    nodes, links, out_links, link_id = {}, [], {}, 0
    for idx, nid in enumerate(order):
        n = api[nid]
        info = oi[n["class_type"]]
        ins, widgets = [], []
        for name, v in input_order(info):
            t, opts, cfg, is_w = kind(v)
            val = n["inputs"].get(name)
            linked = isinstance(val, list) and len(val) == 2 and str(val[0]) in api
            slot = {"name": name, "type": t if not is_w else t, "link": None}
            if is_w:
                slot["widget"] = {"name": name}
                widgets.append(default(t, opts, cfg) if (linked or name not in n["inputs"]) else val)
                if cfg.get("control_after_generate"):
                    widgets.append("fixed")
                if cfg.get("image_upload") or cfg.get("video_upload"):
                    widgets.append("image")
            ins.append(slot)
            if linked:
                link_id += 1
                src = str(val[0])
                st = info_out_type(oi[api[src]["class_type"]], val[1])
                slot["type"] = st if is_w else slot["type"]
                links.append([link_id, int(src), val[1], int(nid), len(ins) - 1, st])
                slot["link"] = link_id
                out_links.setdefault((src, val[1]), []).append(link_id)
        outs = [{"name": nm, "type": ty, "links": None, "slot_index": i}
                for i, (nm, ty) in enumerate(zip(info.get("output_name", info.get("output", [])), info.get("output", [])))]
        rows = max(len(ins), len(outs)) + (len([1 for s in ins if "widget" in s]))
        h = 40 + 24 * rows
        x = depth[nid] * (COL_W + GAP_X)
        y = colY.get(depth[nid], 0)
        colY[depth[nid]] = y + h + GAP_Y
        node = {"id": int(nid), "type": n["class_type"], "pos": [x, y], "size": [COL_W, h], "flags": {}, "order": idx,
                "mode": 2 if n.get("_meta", {}).get("muted") else 0, "inputs": ins, "outputs": outs,
                "properties": {"Node name for S&R": n["class_type"]}, "widgets_values": widgets}
        title = n.get("_meta", {}).get("title")
        if title:
            node["title"] = title
        nodes[nid] = node
    for (src, slot), ids in out_links.items():
        nodes[src]["outputs"][slot]["links"] = ids
    return {"last_node_id": max(int(i) for i in api), "last_link_id": link_id,
            "nodes": [nodes[i] for i in order], "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def info_out_type(info, slot):
    out = info.get("output", [])
    return out[slot] if slot < len(out) else "*"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("api_json")
    ap.add_argument("-o", "--output", default="workflow_editor_full.json")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--object-info", help="fichier enregistré depuis /object_info (au lieu de --url)")
    ap.add_argument("--offline", action="store_true", help="sans ComfyUI: définitions déduites du workflow (moins fiable)")
    a = ap.parse_args()
    api = json.load(open(a.api_json))
    if a.offline:
        oi = offline_object_info(api)
    else:
        oi = json.load(open(a.object_info)) if a.object_info else json.load(urllib.request.urlopen(a.url.rstrip("/") + "/object_info"))
    json.dump(convert(api, oi), open(a.output, "w"), indent=1)
    print(f"{len(api)} nodes -> {a.output}")


if __name__ == "__main__":
    main()
