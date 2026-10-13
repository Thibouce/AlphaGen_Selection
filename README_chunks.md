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
