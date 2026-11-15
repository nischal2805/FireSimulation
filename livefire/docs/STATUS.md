# STATUS

| Field | Value |
| --- | --- |
| Phase | P1 code complete; P2/P3 partly done; work paused for GPU testing |
| Branch | `main` (merged P0+P1+P2-001/002+P3-001/002 and unverified WIP) |
| Last fully tested commit | f00fc65 era: 305 passed, 2 skipped, 1 xfailed (CPU). Later commits (92730bb, 09b316c, merge 585509e) are UNTESTED |
| Gates | Gate 0 approved. Gates 1, 2, 3 NOT passed (need real cameras/GPU and unfinished tasks) |

**Start here: `docs/GPU_TEST_HANDOFF.md`** (state, what is unverified, exact commands, prompt for a new chat).

## Next
- On the GPU machine: run tests, benchmarks, re-run the P3-001 segmentation spike, then continue P2-003..006 and P3-003..006.
