# LatentSync 1.5 Cloud Deployment Template

This template deploys **LatentSync 1.5** (audio-conditioned latent diffusion model for lip sync) on GitHub Actions CPU runners without FastAPI overhead.

---

## Folder Structure

```
backend/cloud/latentsync/
├── .github/
│   └── workflows/
│       └── workflow.yml        # 2-stage GitHub Actions workflow (prepare-assets & run-latentsync)
├── requirements.txt            # Python dependencies (PyTorch CPU, Diffusers, Transformers, Whisper, etc.)
├── inference_utils.py          # Helper functions for threadmaxing, checkpoint retrieval, and input validation
├── server.py                   # Main test entry point executing threadmaxed LatentSync 1.5 inference
└── README.md                   # Service documentation
```

---

## Inputs & Output

- **Inputs**:
  - `clip.mp4`: The presenter video clip (25 FPS).
  - `audio.wav`: The target audio track (16 kHz WAV).
  *(Synthetic sample inputs are automatically generated if files are absent).*

- **Output**:
  - `output.mp4`: Lip-synced output video file.
  - Uploaded directly to GitHub Actions Artifacts (`synced-video`).

---

## Execution Specs

- **Hardware**: Standard GitHub Actions Ubuntu CPU runner.
- **Swap Space**: 12 GB dedicated swap file created automatically to handle diffusion model RAM footprint.
- **Threadmaxing**: Fully enabled across PyTorch, OpenMP, MKL, and OpenBLAS using all available host CPU cores (`nproc`).
