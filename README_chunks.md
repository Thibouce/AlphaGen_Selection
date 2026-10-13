# Chunks automatiques

`build_workflow.py` calcule seul le nombre, la durée et les débuts des chunks LTX d'après la vidéo :

```
python build_workflow.py --video-path C:\plans\plan.mp4 --picks picks.json --overlap 7 -o workflow_api.json
python api_to_editor.py workflow_api.json --url http://127.0.0.1:8188 -o workflow_editor.json
```

- Frames, largeur, hauteur et fps viennent de `ffprobe` (ffmpeg doit être installé), sauf si `--frames/--width/--height/--fps` sont donnés.
- `--chunk` = longueur maximale d'un chunk (49), `--overlap` = recouvrement minimal entre chunks.
- `--chunk-mode fixed` (défaut): chunks de `--chunk` frames; le nombre et les débuts s'adaptent, répartis régulièrement.
- `--chunk-mode shrink`: raccourcit aussi les chunks quand le plan s'y prête (non testé sur la qualité).
- `--queue http://127.0.0.1:8188` envoie directement le graphe à ComfyUI.
- Pour un dossier de plans (cmd): `for %f in (plans\*.mp4) do python build_workflow.py --video-path "%f" --picks "picks\%~nf.json" -o "out\%~nf.json" --queue http://127.0.0.1:8188`

Le calcul des débuts est une réimplémentation, pas le `_engine.chunk_starts` d'origine.

## Node ComfyUI « Alpha Chunk Planner » (travailler dans l'interface)

1. Copiez le dossier `comfyui_chunk_planner/` dans `ComfyUI/custom_nodes/` et redémarrez ComfyUI
   (catégorie *Comfy Compositing*, nom *Alpha Chunk Planner*).
2. Branchez `images` (la plaque), réglez `start` (frame du pointage), `chunk` (longueur max), `overlap`, `mode`.
3. Sorties: `count`, `start`, `last_frame`, `length`, `num_chunks`, `end_a`, `end_b`, `width`, `height`,
   `latent_width`, `latent_height`, `plan` (JSON pour `Stitch`), `summary` (texte: chunks à garder / couper),
   puis `A0..A9` (débuts des `ChunkPad` de la passe A, `start` inclus) et `B0..B9` (passe B, plaque isolée).
4. Reliez-les aux widgets: `ChunkPad` start/length/end, `Stitch.plan`, `Plate`/`TrackMasks` start/count/width/height,
   `SAM3Propagate.end_frame`, `EmptyLTXVLatentVideo` width/height/length, `LTXVEmptyLatentAudio.frames_number`, `Isolate.start`.
5. Coupez (Ctrl+M) les branches de chunk d'index >= `num_chunks` (le `summary` les liste); activez-les au besoin.
   Les emplacements inutilisés reçoivent une valeur valide (le dernier début) pour que les liens restent corrects.

Un workflow déjà câblé est généré avec `--planner 10`:

```
python build_workflow.py --video-path plan.mp4 --picks picks.json --overlap 7 --planner 10 -o workflow_api.json
python api_to_editor.py workflow_api.json --url http://127.0.0.1:8188 -o workflow_editor.json
```
(`workflow_editor_planner_10slots.json` en est un exemple: 133 frames, 3 chunks actifs, 7 emplacements coupés par passe.)

## Chunks inutiles sautés automatiquement (sans Ctrl+M)

- `Alpha Chunk Planner` sort aussi `on_0..on_9` (BOOLEAN): `True` pour les chunks à garder. Ce sont des valeurs, pas des états
  mute/bypass: ComfyUI ne permet pas de couper un node depuis une valeur calculée.
- `Alpha Stitch Auto (lazy)` remplace `ComfyProAlpha_Stitch`: ses entrées `chunk_0..chunk_9` sont « lazy », il ne demande que
  les `num_chunks` premières. Les branches suivantes ne sont jamais exécutées, puis le node se développe en vrai `Stitch`.
  Branchez `plan` et `num_chunks` sur le planificateur.
- Workflow d'exemple: `workflow_editor_planner_auto.json` (`--planner 10 --auto-skip`): 10 branches par passe, aucune coupée.
  Non testé dans ComfyUI: si ça échoue, utilisez la version à branches coupées (`--planner 10`, sans `--auto-skip`).
