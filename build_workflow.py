#!/usr/bin/env python3
"""Génère le workflow ComfyUI (API format) équivalent à l'expansion de ComfyProAlphaGenerator.

Reprend ligne à ligne ComfyProAlphaGenerator.execute() : SAM3 -> Plate/Refine -> LTX alpha-gen par chunks
-> [Isolate -> LTX] -> Final. Le JSON obtenu s'envoie à /prompt ou se charge dans ComfyUI (Load API / drag & drop
sur les versions récentes) pour être modifié node par node.

Exemple :
  python build_workflow.py --video-path C:/plans/plate.mp4 --picks picks.json -o workflow_api.json
  (frames, taille et fps sont lus dans la vidéo; --frames/--width/--height/--fps les remplacent)
"""
import argparse
import json
import math
import os
import subprocess
import urllib.request

from comfyui_chunk_planner import AlphaChunkPlanner
from comfyui_chunk_planner.chunking import legacy_chunk_starts, plan_chunks

PLAN_OUT = {n: i for i, n in enumerate(AlphaChunkPlanner.RETURN_NAMES)}

NONE_PROMPT = "builtin"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def ltx_frames(n):
    """Plus petit 8k+1 >= n (longueur LTX valide)."""
    return 8 * math.ceil((n - 1) / 8) + 1


def chunk_starts(count, L, step):
    """Débuts de chunks recouvrants. ATTENTION: réimplémentation de _engine.chunk_starts (non fournie);
    à remplacer par l'original si le découpage diffère."""
    starts = [0]
    while starts[-1] + L < count:
        starts.append(min(starts[-1] + step, count - L))
    return starts


class Graph:
    def __init__(self):
        self.nodes, self._n, self.label, self.mute = {}, 0, "", False

    def node(self, class_type, **inputs):
        self._n += 1
        nid = str(self._n)
        n = {"class_type": class_type, "inputs": inputs}
        if self.label:
            n["_meta"] = {"title": f"[{self.label}] {class_type}"}
        if self.mute:  # emplacement de chunk inutilisé: coupé (Ctrl+M) dans l'éditeur, absent du graphe envoyé à /prompt
            n.setdefault("_meta", {})["muted"] = True
        self.nodes[nid] = n
        return Out(nid)


class Out:
    def __init__(self, nid):
        self.nid = nid

    def out(self, i=0):
        return [self.nid, i]


def build(a):
    p = json.load(open(a.picks)) if a.picks else {"target": {"frame": 0, "x": 0.5, "y": 0.5}}
    W, H, fps = a.width, a.height, a.fps
    start = int(p["target"]["frame"])
    count = a.frames - start
    if a.render == "preview":
        count = min(count, a.chunk)
    if a.legacy_chunks:
        L = min(a.chunk, ltx_frames(count))
        starts = legacy_chunk_starts(count, L, max(1, L - a.overlap))
    else:
        L, starts = plan_chunks(count, a.chunk, a.overlap, a.chunk_mode == "shrink")
    if len(starts) > 24:
        raise SystemExit(f"{len(starts)} chunks (max 24): baisser --overlap ou couper le plan")
    best = a.quality == "best"
    prefix = a.prefix + ("_preview" if a.render == "preview" else "")
    last = start + count - 1
    slots = a.planner or len(starts)
    if a.planner and a.legacy_chunks:
        raise SystemExit("--planner est incompatible avec --legacy-chunks")
    if len(starts) > slots:
        raise SystemExit(f"{len(starts)} chunks > {slots} emplacements (--planner)")

    g = Graph()
    # --- source de la plaque (remplaçable par Load Plate Sequence, etc.) ---
    g.label = "Source"
    load = g.node("LoadVideo", file=a.video)
    comps = g.node("GetVideoComponents", video=load.out(0))
    src = comps.out(0)

    # --- planificateur: les widgets qui dépendent de la durée sont reliés à ses sorties ---
    pl = None
    if a.planner:
        g.label = "Plan"
        pl = g.node("AlphaChunkPlanner", start=start, chunk=a.chunk, overlap=a.overlap, mode=a.chunk_mode,
                    render="preview chunk" if a.render == "preview" else "full shot", images=src)

    def V(name, literal):
        return [pl.nid, PLAN_OUT[name]] if pl else literal

    g.label = "SAM3"
    vid = g.node("ComfyProAlpha_LoadVideo", images=src, fps=fps)
    sam = g.node("LoadSAM3Model", precision="auto")

    def track(mode, frame, **prompt):
        seg = g.node("SAM3VideoSegmentation", video=vid.out(0), prompt_mode=mode, frame_idx=int(frame),
                     score_threshold=a.sam3_threshold, **prompt)
        prop = g.node("SAM3Propagate", sam3_model_config=sam.out(0), video_state=seg.out(0),
                      start_frame=int(frame), end_frame=V("last_frame", last), direction="forward")
        return g.node("ComfyProAlpha_TrackMasks", masks=prop.out(0), start=V("start", start), count=V("count", count),
                      width=V("width", W), height=V("height", H)).out(0)

    def box_track(b):
        cx, cy, w, h = b["box"]
        box = g.node("SAM3CreateBox", center_x=cx, center_y=cy, width=w, height=h, is_positive=True)
        return track("box", b["frame"], positive_boxes=g.node("SAM3CombineBoxes", box_1=box.out(0)).out(0))

    def point_track(pt):
        neg = pt.get("neg", [])
        pts = g.node("SAM3CombinePoints", **{f"point_{i + 1}": g.node("SAM3CreatePoint", x=x, y=y, is_foreground=True).out(0)
                                            for i, (x, y) in enumerate(pt.get("points") or [[pt["x"], pt["y"]]])})
        kw = {"positive_points": pts.out(0)}
        if neg:
            kw["negative_points"] = g.node("SAM3CombinePoints", **{
                f"point_{i + 1}": g.node("SAM3CreatePoint", x=x, y=y, is_foreground=False).out(0)
                for i, (x, y) in enumerate(neg)}).out(0)
        return track("point", pt["frame"], **kw)

    def union(masks):
        out = None
        for m in masks:
            out = m if out is None else g.node("ComfyProAlpha_MaskMax", a=out, b=m).out(0)
        return out

    target = box_track(p["target"]) if "box" in p["target"] else point_track(p["target"])
    ok = lambda items: [x for x in items if int(x["frame"]) <= last]
    refine_kw = {k: v for k, v in (("adds", union([point_track(x) for x in ok(p.get("adds", []))])),
                                   ("cuts", union([point_track(x) for x in ok(p.get("cuts", []))])),
                                   ("others", union([box_track(x) for x in ok(p.get("others", []))]))) if v is not None}
    g.label = "Refine"
    plate = g.node("ComfyProAlpha_Plate", images=src, start=V("start", start), count=V("count", count))
    ref = g.node("ComfyProAlpha_Refine", plate=plate.out(0), target=target, vitmatte=a.vitmatte,
                 erode=a.erode, dilate=a.dilate, radius=a.stabilize_radius, **refine_kw)
    sel, refined, others = ref.out(0), ref.out(1), ref.out(2)

    # --- pile LTX-2.5 alpha-gen ---
    g.label = "LTX stack"
    if a.ltx_model.lower().endswith(".gguf"):
        unet = g.node("UnetLoaderGGUF", unet_name=a.ltx_model)
    else:
        unet = g.node("UNETLoader", unet_name=a.ltx_model, weight_dtype="default")
    lora = g.node("LTXICLoRALoaderModelOnly", model=unet.out(0), lora_name=a.alpha_lora, strength_model=1.0)
    model = lora.out(0)
    if not a.no_fast_attention:
        model = g.node("ModelAttentionBackend", model=model, attention="comfy kitchen attention").out(0)
    if a.text_encoder == NONE_PROMPT:
        empty = g.node("ComfyProAlpha_EmptyPrompt").out(0)
        cond = g.node("LTXVConditioning", positive=empty, negative=empty, frame_rate=fps)
    else:
        clip = g.node("CLIPLoader", clip_name=a.text_encoder, type="ltxv")
        cond = g.node("LTXVConditioning",
                      positive=g.node("CLIPTextEncode", clip=clip.out(0), text="").out(0),
                      negative=g.node("CLIPTextEncode", clip=clip.out(0), text="").out(0), frame_rate=fps)
    vae = g.node("VAELoader", vae_name=a.video_vae)
    avae = g.node("VAELoader", vae_name=a.audio_vae)
    latent = g.node("EmptyLTXVLatentVideo", width=V("latent_width", W + (-W) % 32), height=V("latent_height", H + (-H) % 32),
                    length=V("length", L), batch_size=1)
    alat = g.node("LTXVEmptyLatentAudio", frames_number=V("length", L), frame_rate=int(round(fps)), batch_size=1, audio_vae=avae.out(0))
    noise = g.node("RandomNoise", noise_seed=a.seed)
    sampler = g.node("KSamplerSelect", sampler_name="euler_ancestral")
    sigmas = g.node("ManualSigmas", sigmas=SIGMAS)
    plan = json.dumps({"n": count, "h": H, "w": W, "chunk": L, "starts": starts})

    def ltx_pass(images, name, offset=0):
        chunks = {}
        for i in range(slots):
            s = starts[min(i, len(starts) - 1)]
            g.label, g.mute = f"LTX {name} chunk {i}", i >= len(starts)
            img = g.node("ComfyProAlpha_ChunkPad", images=images, start=V(f"{name}{i}", offset + s), length=V("length", L),
                         end=V("end_a" if name == "A" else "end_b", offset + count))
            guide = g.node("LTXAddVideoICLoRAGuide", positive=cond.out(0), negative=cond.out(1), vae=vae.out(0),
                           latent=latent.out(0), image=img.out(0), frame_idx=0, strength=1.0,
                           latent_downscale_factor=lora.out(1), crop="disabled", use_tiled_encode=False,
                           tile_size=256, tile_overlap=64)
            av = g.node("LTXVConcatAVLatent", video_latent=guide.out(2), audio_latent=alat.out(0))
            guider = g.node("CFGGuider", model=model, positive=guide.out(0), negative=guide.out(1), cfg=1.0)
            samp = g.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0), sampler=sampler.out(0),
                          sigmas=sigmas.out(0), latent_image=av.out(0))
            sep = g.node("LTXVSeparateAVLatent", av_latent=samp.out(0))
            crop = g.node("LTXVCropGuides", positive=guide.out(0), negative=guide.out(1), latent=sep.out(0))
            dec = g.node("ComfyProAlpha_DecodeAlpha", samples=crop.out(2), vae=vae.out(0))
            chunks[f"chunk_{i}"] = dec.out(0)
        g.label, g.mute = f"LTX {name} stitch", False
        return g.node("ComfyProAlpha_Stitch", plan=V("plan", plan), **chunks).out(0)

    ltx = ltx_pass(src, "A", offset=start)
    final_kw = {}
    if best:
        g.label = "Isolate"
        iso = g.node("ComfyProAlpha_Isolate", plate=src, start=V("start", start), selection=sel, others=others, grow=6, blur=6.0)
        final_kw["ltx_isolated"] = ltx_pass(iso.out(0), "B")
    g.label = "Final"
    final = g.node("ComfyProAlpha_Final", plate=plate.out(0), ltx=ltx, selection=sel, refined=refined,
                   others=others, near_px=a.near_px, filename_prefix=prefix, format=a.format,
                   first_frame=a.first_frame + start, save=True, qc=True, comp=a.comp, **final_kw)
    # sorties visibles: alpha (mask) et comp (image)
    g.node("PreviewImage", images=final.out(1)) if a.comp else None
    return g.nodes


def probe(path):
    """Largeur, hauteur, fps et nombre de frames d'une vidéo (ffprobe)."""
    def run(extra):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *extra, "-show_entries",
                              "stream=width,height,r_frame_rate,nb_frames,nb_read_frames", "-of", "json", path],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)["streams"][0]
    st = run([])
    n = st.get("nb_frames")
    if not n or n == "N/A":
        st = run(["-count_frames"])  # décode tout le flux: plus lent mais exact
        n = st.get("nb_read_frames")
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), int(num) / int(den), int(n)


def strip_muted(nodes):
    """Graphe exécutable: sans les nodes coupés ni les liens qui y mènent (ce que fait l'éditeur à l'envoi)."""
    gone = {k for k, n in nodes.items() if n.get("_meta", {}).get("muted")}
    return {k: {**n, "inputs": {i: v for i, v in n["inputs"].items()
                                if not (isinstance(v, list) and len(v) == 2 and str(v[0]) in gone)}}
            for k, n in nodes.items() if k not in gone}


def queue(nodes, url):
    nodes = strip_muted(nodes)
    req = urllib.request.Request(url.rstrip("/") + "/prompt", json.dumps({"prompt": nodes}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="nom du fichier dans ComfyUI/input (défaut: nom de --video-path)")
    ap.add_argument("--video-path", dest="video_path", help="chemin local de la vidéo: frames, taille et fps sont lus avec ffprobe")
    ap.add_argument("--frames", type=int, help="nombre total de frames (sinon lu avec ffprobe)")
    ap.add_argument("--width", type=int, help="largeur de la plaque (sinon lue avec ffprobe)")
    ap.add_argument("--height", type=int, help="hauteur de la plaque (sinon lue avec ffprobe)")
    ap.add_argument("--fps", type=float, help="fps (sinon lu avec ffprobe, 24 par défaut)")
    ap.add_argument("--first-frame", dest="first_frame", type=int, default=1001)
    ap.add_argument("--picks", help="JSON du picker: {target:{frame,x,y|box}, adds, cuts, others}")
    ap.add_argument("--quality", choices=["best", "fast"], default="best")
    ap.add_argument("--render", choices=["full", "preview"], default="full")
    ap.add_argument("--prefix", default="comfy_pro_alpha/shot")
    ap.add_argument("--format", choices=["png16", "exr"], default="png16")
    ap.add_argument("--ltx-model", default="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    ap.add_argument("--alpha-lora", default="ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors")
    ap.add_argument("--text-encoder", default=NONE_PROMPT, help="'builtin' ou un fichier text_encoders/")
    ap.add_argument("--video-vae", default="ltx-2.5-video-vae-conv-bf16.safetensors")
    ap.add_argument("--audio-vae", default="ltx-2.5-audio-vae-bf16.safetensors")
    ap.add_argument("--vitmatte", default="auto-download")
    ap.add_argument("--chunk", type=int, default=49, help="longueur MAX d'un chunk (8k+1, 9..145)")
    ap.add_argument("--overlap", type=int, default=9, help="recouvrement MIN entre chunks voisins")
    ap.add_argument("--chunk-mode", dest="chunk_mode", choices=["fixed", "shrink"], default="fixed",
                    help="fixed: chunks de --chunk frames, seuls leur nombre et leurs débuts s'adaptent; "
                         "shrink: raccourcit aussi la longueur des chunks (moins de frames calculées)")
    ap.add_argument("--planner", type=int, default=0, metavar="SLOTS",
                    help="graphe avec un node Alpha Chunk Planner et SLOTS branches de chunk par passe (max 10); "
                         "les branches inutiles sont coupées (muted) dans l'éditeur")
    ap.add_argument("--legacy-chunks", dest="legacy_chunks", action="store_true",
                    help="ancien calcul à pas fixe (dernier chunk très recouvrant)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--erode", type=int, default=10)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--near-px", dest="near_px", type=int, default=15)
    ap.add_argument("--stabilize-radius", dest="stabilize_radius", type=int, default=2)
    ap.add_argument("--sam3-threshold", dest="sam3_threshold", type=float, default=0.3)
    ap.add_argument("--no-fast-attention", action="store_true")
    ap.add_argument("--no-comp", dest="comp", action="store_false", help="ne pas construire l'image comp")
    ap.add_argument("-o", "--output", default="workflow_api.json")
    ap.add_argument("--queue", metavar="URL", help="envoie aussi le graphe à ComfyUI (ex. http://127.0.0.1:8188)")
    a = ap.parse_args()
    if a.video_path:
        w, h, fps, n = probe(a.video_path)
        a.width, a.height, a.fps, a.frames = a.width or w, a.height or h, a.fps or fps, a.frames or n
        a.video = a.video or os.path.basename(a.video_path)
        print(f"vidéo: {a.width}x{a.height}, {a.fps:g} fps, {a.frames} frames")
    a.video, a.fps = a.video or "plate.mp4", a.fps or 24.0
    a.width, a.height = a.width or 1920, a.height or 1080
    if not a.frames:
        ap.error("--frames ou --video-path est requis")
    nodes = build(a)
    with open(a.output, "w") as f:
        json.dump(nodes, f, indent=2)
    print(f"{len(nodes)} nodes -> {a.output}")
    if a.queue:
        print("ComfyUI:", queue(nodes, a.queue))


if __name__ == "__main__":
    main()
