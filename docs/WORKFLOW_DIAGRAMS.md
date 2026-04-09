# MPEC-SS Algorithm Workflow Diagrams

**Version:** 1.0.3
**Last Updated:** 2026-03-25

## Overview

This directory contains Mermaid flowchart diagrams documenting the MPEC-SS algorithm workflow. The diagrams are split into three files for mermaid.live compatibility (each under 60 nodes).

## Diagram Files

| File | Description | Nodes |
|------|-------------|-------|
| `diagram_a_overview.mmd` | Top-level overview of all phases | 18 |
| `diagram_b_phase12.mmd` | Phase I + Early Check + Phase II detail | 48 |
| `diagram_c_phase3.mmd` | Phase III + BNLP + Result | 24 |

## How to Render

1. Go to https://mermaid.live
2. Click "Code" tab on the left panel
3. Delete all existing content
4. Copy the contents of ONE `.mmd` file and paste
5. The preview on the right should render instantly

## Parameter Reference (v1.0.3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| t0 | 1.0 | Initial smoothing |
| kappa | 0.5 | Reduction factor |
| eps_tol | 1e-6 | Complementarity tolerance |
| tau | 1e-6 | Sign test tolerance |
| max_outer | 3000 | Max Phase II iterations |
| max_restorations | 50 | Hard cap on restorations |
| restoration_stag_window | 8 | Consecutive no-improve before fail |
| max_adaptive_jumps | 500 | Hard cap on jump regime |
| LPEC_BIACTIVE_THRESHOLD | 15 | Skip cheap LPEC threshold |
| T-floor | 1e-14 | Minimum t_k value |
| Floor stagnation window | 20 | Iterations at floor before exit |

## T-Update Regimes

| Regime | Condition | Formula |
|--------|-----------|---------|
| superlinear | tracking AND improvement > 0.5 | t_new = kappa² × t_k |
| fast | tracking AND improvement > 0.1 | t_new = min(kappa, 0.5) × t_k |
| adaptive_jump | stagnation_count >= 4 | t_new = kappa² × t_k |
| post_stagnation_fast | stagnation_count >= 2 | t_new = min(kappa, 0.5) × t_k |
| slow (default) | otherwise | t_new = kappa × t_k |

**Note:** The current implementation uses `adaptive_jump = kappa² × t_k`.

## Color Legend

| Color | Meaning |
|-------|---------|
| Blue (#dbeafe) | Phase/major component |
| Green (#dcfce7) | Success outcome |
| Red (#fee2e2) | Failure/exit |
| Yellow (#fef9c3) | Decision point |
| Purple (#f3e8ff) | Process step |
| Teal (#ecfdf5) | Bypass/skip path |

## Known Simplifications

- Solver fallback chain (4 strategies) shown as single "NLP + Fallbacks" box
- Restoration cascade strategies (perturb/escape/reg) shown as single box
- Post-loop rescue strategies omitted (gentle t-reduction, partition flip details)
- Multiplier extraction formula simplified (CasADi sign convention not shown)
- Box-MCP three-partition (I3) details in BNLP omitted for clarity
