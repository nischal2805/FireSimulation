# Literature review and novelty verdict (P0-013)

Date of review: 2026-11-14. Time-boxed web search (about 14 searches).

## Method and honesty caveats

- WebFetch could not open any host (arxiv.org, mdpi.com, iclr.cc, mlanthology.org, radiancefields.com,
  ira.lib.polyu.edu.hk all failed DNS: not in the session's network allow-list). So **no full text was read**.
- Evidence level used below:
  - **S (search-confirmed):** the URL was returned by a web search result, and the title, authors, identifier and
    abstract-level summary were returned by the search engine. Existence is confirmed; detailed claims are
    abstract-level only.
  - **unverified:** a specific claim for which no source text was seen.
- Follow-up (P0 gate): re-run with arxiv.org, mdpi.com allowed, and read the three seed papers in full,
  in particular FieryGS runtime and any multi-view or live claims.

## Entries

| # | Citation | URL | What it does | Live / offline | Views | Material-aware | Relation to thesis | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Shen, Tao, Dai et al., "FieryGS: In-the-Wild Fire Synthesis with Physics-Integrated Gaussian Splatting", ICLR 2026, arXiv 2605.00177 | https://arxiv.org/pdf/2605.00177 ; https://iclr.cc/virtual/2026/poster/10006432 | 3DGS scene reconstruction, multimodal-LLM (GPT-4o, zero-shot) infers per-region material, burnability, thermal diffusivity, smoke colour; 256^3 grid volumetric combustion; flame spread, smoke, charring; renderer unified with 3DGS. User controls: ignition, airflow, intensity | Offline-style: reported about 2.37 s/frame on RTX 4090D, under 10 GB (secondary summary); needs a prior 3DGS scan | Multi-view images to build the splat; fire shown in a pre-built 3D scene | Yes (LLM inferred) | Closest prior art on material-aware fire in a real captured scene. Differs: offline scan, not live, about 2 s/frame, no live camera compositing | S (runtime figure unverified, from aggregator) |
| 2 | Driessen, Rijsdijk, Pirk, Palubicki, Michels, Weinmann, "WildFireGS: Physics-Based Wildfire Simulation in Large-Scale Semantics-Enriched Gaussian Splatting Forest Scenes", arXiv 2608.11100 | https://arxiv.org/pdf/2608.11100 | Drone-imagery GS reconstruction, semantic vegetation and fuel classes on Gaussians, particle-based combustion (ignition, heat transfer, spread) directly on Gaussians, rain cooling | Offline; no real-time claim seen | Aerial multi-view imagery (reconstruction) | Semantic fuel classes (vegetation) | Related: simulation on reconstructed scene, but outdoor wildfire scale, offline | S |
| 3 | Wong, Wong, Das, Yuen, Cheung, "Transferring Fire Modelling Sciences into Augmented Reality: A Realistic and Safe Reconstructed Fire Scenario", Fire 8(4):132, 2025, doi 10.3390/fire8040132 | https://www.mdpi.com/2571-6255/8/4/132 ; https://ira.lib.polyu.edu.hk/bitstream/10397/117858/1/fire-08-00132.pdf | AR platform: geometry extraction, scientific fire dynamics simulation, data processing, visualisation in real settings, for evacuee behaviour studies (survey: 77% preferred AR) | Described as "real-time" in abstract, but pipeline is geometry extraction then simulation then AR; whether the sim is precomputed is unverified | Single AR headset view (unverified) | Unverified (apparently no per-object materials inferred from live frames) | Related: AR fire for evacuation studies. Does not appear to be live acquisition from two cameras with one shared state | S (live vs precomputed unverified) |
| 4 | Wagner, Rao, Wrede, Pirk, Xiao, "Fire as a Service: Augmenting Robot Simulators with Thermally and Visually Accurate Fire Dynamics", arXiv 2603.19063, Mar 2026 | https://arxiv.org/pdf/2603.19063.pdf ; https://www.alphaxiv.org/abs/2603.19063 | Asynchronous fire co-simulation for robot simulators (Gazebo, Isaac Sim, MuJoCo) via ROS 2; receives camera params, poses, RGB and depth; GPU hybrid solver Fire-X; renders fire from the robot camera, depth-based occlusion, alpha-blends onto RGB; about 112 ms round-trip (reported); thermal dose maps | Near-real-time, streaming, but inputs come from a simulator, not real cameras | Per robot camera (per-view render) | Fire-X multi-species thermochemistry; material source for scene unverified | Closest on streaming depth + RGB to fire render and composite; differs: virtual scene, not real live cameras, no acquisition, no shared state across two real views | S (summary via alphaxiv/search; material handling unverified) |
| 5 | "Physics-driven Fire Modeling from Multi-view Images", arXiv 1804.05261 (2018) | https://arxiv.org/pdf/1804.05261 | Estimates fire temperature and fuel density from multi-view RGB of real flames | Offline | Multi-view | No | Older, inverse direction (reconstruct real fire), not synthesis on a scene | S (authors, venue unverified) |
| 6 | Driessen et al. (TU Delft thesis), "A Physics-Based Forest Fire Simulation using Semantic Gaussian Splatting" | https://repository.tudelft.nl/record/uuid:008cbe35-dfe1-494b-9efe-3e0856b95204 | Thesis precursor to WildFireGS | Offline | Aerial | Semantic fuel | Duplicate of #2 for our purposes; noted limitations: noisy data, semantic classification, scalability | S (author list of thesis unverified) |
| 7 | Stony Brook, "Real-Time Fire and Smoke" (Lattice Boltzmann on GPU, textured splats) | https://www3.cs.stonybrook.edu/~mueller/papers/Fire.pdf | GPU LBM fire and smoke with splat rendering, real-time | Real-time | n/a (synthetic scene) | No | Classic precedent for real-time GPU fire; no camera input. Authors, year unverified | S (bibliographic details unverified) |
| 8 | Live compositing for virtual production, e.g. Assimilate Live FX | https://www.shootonline.com/spw/assimilate-announces-open-beta-new-live-fx-live-compositing-virtual-productions/ | Commercial live compositing of camera feeds with digital backgrounds for on-set previs. Vendor claims only | Live | Multi-camera feeds | No (no fire physics) | Product landscape: no physics fire from acquired scene seen | S (vendor marketing) |
| 9 | US 6,500,008 / US 6,809,744, AR firefighter training patents | https://patents.google.com/patent/US6809744 | Head-worn camera plus CG fire (particle systems for flame and plume), luminance-key or alpha blend, tracker prediction for delay | Live | Single | No | Old art for live camera plus CG fire overlay; no scene materials | S (patent numbers as returned by search; only 6809744 URL is a patent page) |
| 10 | Utrecht Univ. thesis on real-time occlusion-aware mixed reality for broadcast | https://ics-gmt.science.uu.nl/sites/default/files/2026-04/RealTimeOcclusionAwareMixedReality.pdf | Per-pixel depth ordering of virtual objects into live video (about 35 ms pipeline) | Live | Single | n/a | Supports the compositing and occlusion approach; not fire | S (title and author unverified) |

Background works named in PRD section 1 (Stam 1999; Fedkiw, Stam, Jensen 2001; Nguyen, Fedkiw, Jensen 2002) were
not searched in this pass; they are well-known references, URLs not opened: **unverified here**.

Seed claims in PRD section 1 checked:

- FieryGS "appears offline": consistent with search evidence (needs a 3DGS scan; about 2.37 s/frame reported). Confirmed at abstract level.
- WildFireGS "wildfire simulation on semantic Gaussian-splat forest scenes": confirmed.
- Wong et al. "AR fire-dynamics platform for evacuation studies": confirmed. The PRD wording is accurate; whether the simulation is live is unverified.

## Verdict

**Does a live, two-view, material-aware fire previs system already exist?** Not found. Within this limited search
(abstract-level evidence, no full texts, about 14 queries), no work combines all of: (a) live camera streams with no
offline scan, (b) scene geometry and materials acquired from those live frames, (c) one 3D fire state shared across
two real views, (d) interactive rates on a consumer laptop GPU.

What exists is partial:

| Capability | Nearest prior art | Gap vs thesis |
|---|---|---|
| Material-aware fire in a real captured scene | FieryGS | Offline 3DGS scan, about 2 s/frame on a desktop GPU, not live |
| Streaming RGB-D to fire render and alpha composite with occlusion | FaaS | Simulator-sourced frames, no real acquisition, per-camera render |
| Live AR fire overlay | Wong et al. 2025; firefighter AR patents | Not material-aware from live frames; not two-view shared state (unverified) |
| Real-time GPU fire | Classic LBM and Eulerian GPU work | No scene acquisition from video |

**Conclusion: the thesis stands, but the novelty claim must be phrased as a combination, not as any single element.**
Individual ingredients (LLM or segmentation material inference, depth-based occlusion compositing, GPU gas solver,
streamed RGB-D fire co-simulation) each have prior art. The defensible novelty is the integration: live
two-camera acquisition (about 15 s) to material table to GPU surface-plus-gas simulation, one shared fire state
composited onto both live views at interactive rates with measured latency.

**Recommended narrowing and wording (record in PRD change control if adopted):**

- Do not claim "first fire on a reconstructed real scene" or "first material-aware fire" (FieryGS).
- Do not claim "first real-time fire compositing with depth occlusion" (FaaS reports about 112 ms in a robot-simulator setting).
- Claim: "live-acquired (no offline scan) material-aware fire shared across two real camera views on a laptop GPU, with measured latency".
- Position FieryGS as the quality and offline baseline, FaaS as the latency and compositing reference.

## Residual risk and follow-ups

1. Re-read all three seed papers and FaaS in full once arxiv.org and mdpi.com are reachable. Specifically check: any real-time or
   live mode in FieryGS or its project page (https://pku-vcl-geometry.github.io/FieryGS/, mentioned in search output, not opened);
   whether Wong et al. runs a live simulation and with how many views.
2. Searches not done: Stam, Fedkiw and Nguyen background URLs; NeRF/GS fire editing before 2026; multi-camera
   scene-flow or live-GS (streaming Gaussian splatting) work that could undercut "no offline scan". A live-GS
   or fast feed-forward reconstruction paper combined with FieryGS-style fire would weaken the novelty; re-check before demo.
3. Search tool covered only US web results; non-English and non-indexed work may be missed.
