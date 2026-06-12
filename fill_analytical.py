"""
Fill missing (model, device, ctx_K, batch) rows in paper_comprehensive.csv
by extrapolating from existing simulated data.

Strategy for dense_tpot:
  - Dense TPOT is linear in context length once flash-bound.
  - For each (model, device, batch), fit dense_tpot = a * ctx_K + b on existing points
    and extrapolate to missing ctx_K values.

Strategy for sparse_tpot:
  - Sparse TPOT is roughly flat (compute-bound) until the 10% flash fraction dominates.
  - Use the last available sparse_tpot for that (model, device, batch) as a floor,
    then extrapolate using a 10% slope of the dense fit if sparse becomes flash-bound.

All filled rows are tagged source=analytical.
"""

import csv
import sys
from collections import defaultdict

INPUT  = "results/paper_comprehensive.csv"
OUTPUT = "results/paper_comprehensive_filled.csv"

ALL_DEVICES = ['a100', 'a40', 'h100']
ALL_CTX_K   = [4, 8, 16, 32, 64, 128, 256, 512]
ALL_BATCHES = [4, 8, 16, 32, 64, 128]

with open(INPUT) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    existing_rows = list(reader)

existing_keys = {
    (r['model'], r['device'], int(r['ctx_K']), int(r['batch']))
    for r in existing_rows
}

# Build per-(model, device, batch) sorted lists of (ctx_K, dense_tpot, sparse_tpot)
data = defaultdict(list)
for r in existing_rows:
    key = (r['model'], r['device'], int(r['batch']))
    data[key].append((
        int(r['ctx_K']),
        float(r['dense_tpot_p50_ms']),
        float(r['sparse_tpot_p50_ms']),
    ))
for key in data:
    data[key].sort()


def fit_linear(xs, ys):
    """Least-squares linear fit y = a*x + b. Returns (a, b)."""
    n = len(xs)
    if n == 1:
        return 0.0, ys[0]
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    if denom == 0:
        return 0.0, sy/n
    a = (n*sxy - sx*sy) / denom
    b = (sy - a*sx) / n
    return a, b


def extrapolate_dense(pts, target_ctx_K):
    """
    Extrapolate dense_tpot to target_ctx_K.
    Uses only flash-bound points (where stall > 0, i.e., dense > sparse)
    for the linear fit. Falls back to all points if none are flash-bound.
    """
    flash_pts = [(ctx, d) for ctx, d, s in pts if d > s * 1.01]
    if len(flash_pts) >= 2:
        xs, ys = zip(*flash_pts)
        a, b = fit_linear(list(xs), list(ys))
        return max(a * target_ctx_K + b, pts[-1][1])
    elif len(flash_pts) == 1:
        # Scale from the one flash-bound point using slope from all pts
        xs, ys = zip(*[(c, d) for c, d, _ in pts])
        a, b = fit_linear(list(xs), list(ys))
        if a > 0:
            return max(a * target_ctx_K + b, pts[-1][1])
        return flash_pts[0][1] * target_ctx_K / flash_pts[0][0]
    else:
        # Fully compute-bound across all existing points — extrapolate flat
        # (dense = sparse = T_compute, no flash stall even at target)
        xs, ys = zip(*[(c, d) for c, d, _ in pts])
        a, b = fit_linear(list(xs), list(ys))
        return max(a * target_ctx_K + b, pts[-1][1])


def extrapolate_sparse(pts, target_ctx_K, dense_extrap):
    """
    Extrapolate sparse_tpot to target_ctx_K.
    Sparse = compute-bound floor (flat) unless sparse trend shows flash growth.
    The sparse TPOT cannot exceed dense_extrap.
    """
    # Compute-bound floor: last sparse value is representative T_compute
    compute_floor = pts[-1][2]

    # Check if sparse shows any flash growth
    sparse_flash_pts = [(ctx, s) for ctx, _, s in pts
                        if s > pts[0][2] * 1.05]  # >5% above minimum
    if len(sparse_flash_pts) >= 2:
        xs, ys = zip(*sparse_flash_pts)
        a, b = fit_linear(list(xs), list(ys))
        extrap = max(a * target_ctx_K + b, compute_floor)
    else:
        # Flat: sparse stays at compute floor
        extrap = compute_floor

    return min(extrap, dense_extrap)


new_rows = []
n_filled = 0

all_models = sorted(set(r['model'] for r in existing_rows))

for model in all_models:
    for device in ALL_DEVICES:
        for batch in ALL_BATCHES:
            pts = data.get((model, device, batch), [])
            if not pts:
                continue  # no existing data for this (model, device, batch) at all

            for ctx_K in ALL_CTX_K:
                if (model, device, ctx_K, batch) in existing_keys:
                    continue

                dense_extrap  = extrapolate_dense(pts, ctx_K)
                sparse_extrap = extrapolate_sparse(pts, ctx_K, dense_extrap)

                # Clamp: sparse <= dense, both > 0
                dense_extrap  = max(dense_extrap, 0.1)
                sparse_extrap = max(sparse_extrap, 0.1)
                sparse_extrap = min(sparse_extrap, dense_extrap)

                speedup = dense_extrap / sparse_extrap

                # Stall: proportional to dense excess over compute floor
                t_compute = pts[-1][2]  # last sparse as T_compute proxy
                dense_stall  = max(0.0, dense_extrap  - t_compute)
                sparse_stall = max(0.0, sparse_extrap - t_compute)

                # Throughput tokens/s
                dense_tput  = batch / (dense_extrap  / 1000.0)
                sparse_tput = batch / (sparse_extrap / 1000.0)

                # p99 ratio from existing data
                p99_rows = [(r['dense_tpot_p99_ms'], r['dense_tpot_p50_ms'])
                            for r in existing_rows
                            if r['model'] == model and r['device'] == device
                            and int(r['batch']) == batch
                            and float(r['dense_tpot_p50_ms']) > 0]
                if p99_rows:
                    ratios = [float(p99)/float(p50)
                              for p99, p50 in p99_rows
                              if float(p50) > 0]
                    p99_factor = sum(ratios) / len(ratios)
                else:
                    p99_factor = 1.3

                new_rows.append({
                    'model':                     model,
                    'device':                    device,
                    'ctx_K':                     ctx_K,
                    'batch':                     batch,
                    'dense_tpot_p50_ms':         f"{dense_extrap:.4f}",
                    'dense_tpot_p99_ms':         f"{dense_extrap * p99_factor:.4f}",
                    'sparse_tpot_p50_ms':        f"{sparse_extrap:.4f}",
                    'sparse_tpot_p99_ms':        f"{sparse_extrap * p99_factor:.4f}",
                    'speedup_dense_over_sparse': f"{speedup:.4f}",
                    'dense_hbf_stall_ms':        f"{dense_stall:.4f}",
                    'sparse_hbf_stall_ms':       f"{sparse_stall:.4f}",
                    'dense_throughput_tok_s':     f"{dense_tput:.2f}",
                    'sparse_throughput_tok_s':    f"{sparse_tput:.2f}",
                    'source':                    'analytical',
                })
                n_filled += 1

print(f"Existing rows:  {len(existing_rows)}")
print(f"Pass-1 filled:  {n_filled}")

# ── Pass 2: fill remaining (model, device, ctx_K, batch) with no base data
# Happens for Qwen-72B/bs=64,128 and internlm-20b/bs=128 where the largest
# simulated batch is lower. Scale from the nearest available (ctx_K, lower_batch).
filled_so_far = existing_keys | {
    (r['model'], r['device'], int(r['ctx_K']), int(r['batch'])) for r in new_rows
}

# Build lookup across simulated + pass-1 filled
all_so_far = existing_rows + new_rows
lookup_so_far = {
    (r['model'], r['device'], int(r['ctx_K']), int(r['batch'])): r
    for r in all_so_far
}

p2_rows = []
for model in all_models:
    for device in ALL_DEVICES:
        for ctx_K in ALL_CTX_K:
            for batch in ALL_BATCHES:
                if (model, device, ctx_K, batch) in filled_so_far:
                    continue

                # Find the largest available batch for same (model, device, ctx_K)
                ref_batch = None
                for b in sorted(ALL_BATCHES, reverse=True):
                    if (model, device, ctx_K, b) in filled_so_far:
                        ref_batch = b
                        break

                if ref_batch is None:
                    # No data at same ctx — find nearest ctx
                    for ctx_alt in sorted(ALL_CTX_K, reverse=True):
                        for b in sorted(ALL_BATCHES, reverse=True):
                            if (model, device, ctx_alt, b) in filled_so_far:
                                ref_batch = b
                                ref_ctx   = ctx_alt
                                break
                        if ref_batch is not None:
                            break
                    if ref_batch is None:
                        continue
                    ref_row = lookup_so_far[(model, device, ref_ctx, ref_batch)]
                    scale_dense  = batch / ref_batch
                    scale_sparse = batch / ref_batch
                else:
                    ref_row = lookup_so_far[(model, device, ctx_K, ref_batch)]

                    # Compute batch-scaling factor from available data
                    # Use ratio of (ref_batch) / (next lower batch) to extrapolate
                    lower_batches = [b for b in ALL_BATCHES
                                     if b < ref_batch
                                     and (model, device, ctx_K, b) in filled_so_far]
                    if lower_batches:
                        lower_batch = max(lower_batches)
                        lower_row   = lookup_so_far[(model, device, ctx_K, lower_batch)]
                        scale_dense = (float(ref_row['dense_tpot_p50_ms']) /
                                       float(lower_row['dense_tpot_p50_ms']))
                        scale_dense = max(1.0, scale_dense) ** (
                            (batch / ref_batch) / (ref_batch / lower_batch))
                        scale_sparse = (float(ref_row['sparse_tpot_p50_ms']) /
                                        float(lower_row['sparse_tpot_p50_ms']))
                        scale_sparse = max(1.0, scale_sparse) ** (
                            (batch / ref_batch) / (ref_batch / lower_batch))
                    else:
                        scale_dense  = batch / ref_batch
                        scale_sparse = batch / ref_batch

                dense_extrap  = float(ref_row['dense_tpot_p50_ms'])  * scale_dense
                sparse_extrap = float(ref_row['sparse_tpot_p50_ms']) * scale_sparse
                sparse_extrap = min(sparse_extrap, dense_extrap)
                speedup = dense_extrap / max(sparse_extrap, 0.001)
                t_compute = float(ref_row['sparse_tpot_p50_ms'])
                dense_stall  = max(0.0, dense_extrap  - t_compute)
                sparse_stall = max(0.0, sparse_extrap - t_compute)
                dense_tput  = batch / (dense_extrap  / 1000.0)
                sparse_tput = batch / (sparse_extrap / 1000.0)
                p99_factor  = float(ref_row['dense_tpot_p99_ms']) / max(float(ref_row['dense_tpot_p50_ms']), 0.001)

                p2_rows.append({
                    'model':                     model,
                    'device':                    device,
                    'ctx_K':                     ctx_K,
                    'batch':                     batch,
                    'dense_tpot_p50_ms':         f"{dense_extrap:.4f}",
                    'dense_tpot_p99_ms':         f"{dense_extrap * p99_factor:.4f}",
                    'sparse_tpot_p50_ms':        f"{sparse_extrap:.4f}",
                    'sparse_tpot_p99_ms':        f"{sparse_extrap * p99_factor:.4f}",
                    'speedup_dense_over_sparse': f"{speedup:.4f}",
                    'dense_hbf_stall_ms':        f"{dense_stall:.4f}",
                    'sparse_hbf_stall_ms':       f"{sparse_stall:.4f}",
                    'dense_throughput_tok_s':     f"{dense_tput:.2f}",
                    'sparse_throughput_tok_s':    f"{sparse_tput:.2f}",
                    'source':                    'analytical',
                })

new_rows.extend(p2_rows)
n_filled += len(p2_rows)
print(f"Pass-2 filled:  {len(p2_rows)}")
print(f"Total filled:   {n_filled}")
print(f"Total rows:     {len(existing_rows) + n_filled}")

# Tag existing rows
for r in existing_rows:
    r['source'] = 'simulated'

all_rows = existing_rows + new_rows
all_rows.sort(key=lambda r: (r['model'], r['device'], int(r['ctx_K']), int(r['batch'])))

out_fieldnames = fieldnames + ['source']
with open(OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=out_fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Written to {OUTPUT}")

# Spot-check: show CodeLlama/a100/bs=8 full sequence
print("\nCodeLlama-34b / a100 / bs=8 (simulated + extrapolated):")
for r in all_rows:
    if (r['model'] == 'CodeLlama-34b-Instruct-hf'
            and r['device'] == 'a100'
            and int(r['batch']) == 8):
        tag = '*' if r['source'] == 'analytical' else ' '
        print(f"  {tag} ctx={int(r['ctx_K']):>4}K  "
              f"dense={float(r['dense_tpot_p50_ms']):8.2f}ms  "
              f"sparse={float(r['sparse_tpot_p50_ms']):8.2f}ms  "
              f"speedup={float(r['speedup_dense_over_sparse']):.2f}x  "
              f"[{r['source']}]")
