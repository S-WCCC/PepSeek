"""Training, fixed validation selection, resumable checkpoints and locked evaluation."""
import math
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import Dataset, collate, positive_pairs
from .features import FeatureStore
from .io import read_json, write_json, write_csv, save_torch, load_torch, sha256, digest, provenance
from .model import UnifiedPepSeek, ModelConfig, score_matrix
from .losses import multitask_loss, contrastive_loss
from .metrics import binary_metrics, regression_metrics, observed_region_metrics


def load_model(checkpoint, device="cpu"):
    state = load_torch(checkpoint)
    model = UnifiedPepSeek(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def batch_loss(model, dataset, rows, store, device, config, known, training=True):
    inputs = collate(dataset, rows, store, device)
    labels = [dataset.labels(r) for r in rows]
    detail = any(labels)
    distances = any("distance_bins" in lab for lab in labels)
    # Encodings are shared between positive and crossed pairs in this training batch.
    p, q = model.encode(inputs["protein"], "protein"), model.encode(inputs["peptide"], "peptide")
    outputs = model.interact(p, q, detail=detail, distance=distances)
    loss, terms = multitask_loss(outputs, rows, labels, model.cfg.endpoints, config.get("loss_weights"))
    weight = config.get("loss_weights", {}).get("contrastive", 1.)
    if weight:
        mask = torch.tensor([[(r["protein_id"], t["peptide_id"]) in known for t in rows] for r in rows], device=device)
        scores = score_matrix(model, p, q, config.get("pair_chunk", 8))
        con = contrastive_loss(scores, mask, config.get("temperature", .1))
        loss = loss + weight * con
        terms["contrastive"] = con
    if not terms:
        raise ValueError("Batch has no usable supervision")
    return loss, terms, outputs


@torch.no_grad()
def validate(model, dataset, rows, store, device, config):
    if not rows:
        raise ValueError("A nonempty external valid partition is required")
    model.eval()
    total, n = 0., 0
    known = positive_pairs(dataset.split("train") + rows)
    for start in range(0, len(rows), config["batch_size"]):
        batch = rows[start:start + config["batch_size"]]
        loss, _, _ = batch_loss(model, dataset, batch, store, device, config, known, False)
        total += float(loss) * len(batch)
        n += len(batch)
    return total / n


def _setup_device(requested):
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if requested.startswith("cuda"):
        local = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local}" if world > 1 else requested
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    return device, rank, world


def train(manifest, feature_db, model_config, train_config, output, device="cpu", resume=False):
    config = read_json(train_config)
    for key in ("epochs", "batch_size", "max_steps", "patience"):
        if int(config.get(key, 0)) < 1:
            raise ValueError(f"Training configuration needs positive {key}")
    device, rank, world = _setup_device(device)
    threads = int(config.get("cpu_threads", 4))
    torch.set_num_threads(threads)
    seed = int(config.get("seed", 17))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset = Dataset(manifest)
    store = FeatureStore(feature_db, readonly=True)
    seal = dataset.seal()
    cfg = ModelConfig(**read_json(model_config))
    if cfg.esm_dim != store.config["dim"]:
        raise ValueError("Model / ESM dimension mismatch")
    if store.config["backend"].startswith("toy") and not config.get("allow_toy_features", False):
        raise ValueError("Toy features are restricted to explicit software acceptance runs")
    # Verify completeness before allocating a training run.
    for r in dataset.split("train") + dataset.split("valid"):
        for side in ("protein", "peptide"):
            if not store.contains(store.key(dataset.entities[side][r[side + "_id"]], side)):
                raise ValueError("Features are incomplete; run prepare")
    out = Path(output)
    if (out / "SELECTION_SEAL.json").exists():
        raise ValueError("Run already finalized; use a new directory for a new experiment")
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("Output is not empty; explicit --resume is required")
    out.mkdir(parents=True, exist_ok=True)
    model = UnifiedPepSeek(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=config.get("lr_patience", 3))
    use_amp = bool(config.get("bf16", False)) and device.startswith("cuda")
    if use_amp and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requested on unsupported device")
    start_epoch, step, best, bad_epochs = 0, 0, math.inf, 0
    history = []
    coverage = {"binding": 0, "affinity": {ep: 0 for ep in cfg.endpoints},
                "contact": 0, "protein_interface": 0, "peptide_interface": 0, "distance_bins": 0, "regions": 0}
    for row in dataset.split("train"):
        coverage["binding"] += int(row.get("binding") is not None)
        for a in row.get("affinity", []):
            if a["endpoint"] not in coverage["affinity"]:
                raise ValueError("Training endpoint absent from model configuration")
            coverage["affinity"][a["endpoint"]] += 1
        for key, array in dataset.labels(row).items():
            observed = (array >= 0) if key == "distance_bins" else np.isfinite(array)
            coverage[key] += int(observed.any())
    run_identity = {"dataset_sha256": seal["sha256"], "feature_recipe": store.config,
                    "model_config": model.configuration(), "train_config": config, "world_size": world,
                    "train_supervision_coverage": coverage,
                    "contrastive_scope": "local rank batch; accumulation does not add negatives",
                    "provenance": provenance()}
    if resume:
        saved = load_torch(out / "last.pt")
        for k in ("dataset_sha256", "feature_recipe", "model_config", "train_config", "world_size"):
            if saved["identity"][k] != run_identity[k]:
                raise ValueError(f"Resume mismatch: {k}")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, step, best, bad_epochs = saved["epoch"] + 1, saved["step"], saved["best"], saved["bad_epochs"]
        history = read_json(out / "history.json")
    if rank == 0:
        write_json(out / "run.json", run_identity)
        write_json(out / "DATA_SEAL.json", seal)
    stopped = [False]
    def handle_signal(signum, frame):
        stopped[0] = True
    old = {s: signal.signal(s, handle_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    began = time.monotonic()
    train_rows, valid_rows = dataset.split("train"), dataset.split("valid")
    if not train_rows:
        raise ValueError("Empty train partition")
    known = positive_pairs(train_rows)
    terminal = "epochs_completed"
    def checkpoint(epoch):
        return {"model_config": model.configuration(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "step": step, "best": best, "bad_epochs": bad_epochs,
                "identity": run_identity}
    try:
        # Epoch zero is an eligible, auditable fallback.
        if not resume:
            value = validate(model, dataset, valid_rows, store, device, config) if rank == 0 else 0.
            scalar = torch.tensor(value, device=device)
            if world > 1:
                dist.broadcast(scalar, 0)
            best = float(scalar)
            if not math.isfinite(best):
                raise ValueError("Nonfinite epoch-zero validation loss")
            history.append({"epoch": -1, "step": 0, "train_loss": None, "valid_loss": best, "selected": True})
            if rank == 0:
                save_torch(out / "best.pt", checkpoint(-1))
        for epoch in range(start_epoch, config["epochs"]):
            # Epoch-based RNG yields deterministic epoch-boundary restart at fixed world size.
            torch.manual_seed(seed + epoch * world + rank)
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows)).tolist()
            width = config["batch_size"] * world
            pad = (-len(order)) % width
            if pad:
                order += (order * (pad // len(order) + 1))[:pad]
            epoch_loss, batches = 0., 0
            for start in range(0, len(order), width):
                flag = stopped[0] or step >= config["max_steps"] or (config.get("max_seconds") is not None and time.monotonic() - began >= config["max_seconds"])
                stop_tensor = torch.tensor(int(flag), device=device)
                if world > 1:
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                if stop_tensor.item():
                    terminal = "interrupted" if stopped[0] else "budget_reached"
                    break
                ids = order[start + rank * config["batch_size"]:start + (rank + 1) * config["batch_size"]]
                rows = [train_rows[j] for j in ids]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _, _ = batch_loss(model, dataset, rows, store, device, config, known)
                finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                # Explicit data-parallel gradient averaging supports unused, masked heads.
                if world > 1:
                    for param in model.parameters():
                        used = torch.tensor(int(param.grad is not None), device=device)
                        dist.all_reduce(used, op=dist.ReduceOp.MAX)
                        if not used.item():
                            continue
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        dist.all_reduce(param.grad)
                        param.grad.div_(world)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.), error_if_nonfinite=True)
                optimizer.step()
                step += 1; batches += 1; epoch_loss += float(loss.detach())
            val = validate(model, dataset, valid_rows, store, device, config) if rank == 0 else 0.
            scalar = torch.tensor(val, device=device)
            if world > 1:
                dist.broadcast(scalar, 0)
            val = float(scalar)
            if not math.isfinite(val):
                raise FloatingPointError("Nonfinite validation loss")
            improved = val < best - config.get("min_delta", 1e-6)
            if improved:
                best, bad_epochs = val, 0
            else:
                bad_epochs += 1
            scheduler.step(val)
            hist = {"epoch": epoch, "step": step, "train_loss": epoch_loss / max(batches, 1),
                    "valid_loss": val, "selected": improved}
            history.append(hist)
            if rank == 0:
                if improved:
                    save_torch(out / "best.pt", checkpoint(epoch))
                save_torch(out / "last.pt", checkpoint(epoch))
                write_json(out / "history.json", history); write_csv(out / "history.csv", history)
                print(hist, flush=True)
            if terminal != "epochs_completed":
                break
            if bad_epochs >= config["patience"]:
                terminal = "early_stopped"
                break
        if rank == 0:
            chosen = load_torch(out / "best.pt")
            record = {"status": terminal, "steps": step, "selection": "minimum valid total loss; ties retain earlier checkpoint; epoch zero eligible",
                      "selected_epoch": chosen["epoch"], "validation_loss": best,
                      "checkpoint_sha256": sha256(out / "best.pt"), "dataset_sha256": seal["sha256"],
                      "feature_recipe_sha256": digest(store.config), "test_labels_used": False,
                      "software_demo": store.config["backend"].startswith("toy"), "provenance": provenance()}
            if terminal == "interrupted":
                write_json(out / "INTERRUPTED.json", record)
            else:
                write_json(out / "SELECTION_SEAL.json", record)
        if world > 1:
            dist.barrier()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        store.close()
        if world > 1:
            dist.destroy_process_group()
    return {"output": str(out), "status": terminal, "steps": step}


@torch.inference_mode()
def evaluate(manifest, features, checkpoint, output, split="test", device="cpu", batch_size=4):
    if split not in ("train", "valid", "test"):
        raise ValueError("Invalid evaluation split")
    ds = Dataset(manifest); seal = ds.seal()
    model, state = load_model(checkpoint, device)
    selection = read_json(Path(checkpoint).parent / "SELECTION_SEAL.json")
    if selection["checkpoint_sha256"] != sha256(checkpoint) or selection["dataset_sha256"] != seal["sha256"]:
        raise ValueError("Checkpoint/data seal mismatch")
    store = FeatureStore(features, readonly=True)
    if state["identity"]["feature_recipe"] != store.config:
        raise ValueError("Evaluation features differ from training")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = ds.split(split)
    if not rows:
        raise ValueError("Empty evaluation partition")
    predictions, affinity_rows, structural = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        lab = [ds.labels(r) for r in batch]
        inputs = collate(ds, batch, store, device)
        outputs = model(inputs, detail=any(lab), distance=any("distance_bins" in x for x in lab))
        for i, (r, labels) in enumerate(zip(batch, lab)):
            predictions.append({"id": r["id"], "protein_id": r["protein_id"], "peptide_id": r["peptide_id"],
                                "binding": r.get("binding"), "score": float(outputs["score"][i])})
            for a in r.get("affinity", []):
                j = model.cfg.endpoints.index(a["endpoint"])
                affinity_rows.append({"id": r["id"], "endpoint": a["endpoint"], "lower": a.get("lower"),
                                      "upper": a.get("upper"), "prediction": float(outputs["affinity"][i, j])})
            arrays = {}
            for key, head in (("contact", "contact_logits"), ("protein_interface", "protein_interface_logits"), ("peptide_interface", "peptide_interface_logits")):
                if key in labels:
                    shape = labels[key].shape
                    pred = outputs[head][i][tuple(slice(0, n) for n in shape)].float().cpu().numpy()
                    arrays[key + "_probability"] = 1 / (1 + np.exp(-np.clip(pred, -80, 80)))
                    structural.append({"id": r["id"], "head": key, **binary_metrics(labels[key].ravel(), pred.ravel())})
            if "distance_bins" in labels:
                target = labels["distance_bins"]
                logits = outputs["distance_log_probs"][i, :target.shape[0], :target.shape[1]].cpu().numpy()
                prediction = logits.argmax(-1); valid = target >= 0
                arrays["distance_bin_prediction"] = prediction
                f1s = []
                for label in range(model.cfg.distance_bins):
                    tp = ((prediction == label) & (target == label) & valid).sum()
                    denom = ((prediction == label) & valid).sum() + ((target == label) & valid).sum()
                    f1s.append(float(2 * tp / denom) if denom else 0.)
                structural.append({"id": r["id"], "head": "distance", "n": int(valid.sum()),
                                   "accuracy": float((prediction[valid] == target[valid]).mean()) if valid.any() else None,
                                   "macro_f1": float(np.mean(f1s)) if valid.any() else None})
            if labels:
                lp = len(ds.entities["protein"][r["protein_id"]]["sequence"])
                arrays["region_probability"] = outputs["region_membership_logits"][i, :, :lp].sigmoid().cpu().numpy()
                if "regions" in labels:
                    structural.append({"id": r["id"], "head": "regions", **observed_region_metrics(arrays["region_probability"], labels["regions"])})
            if arrays:
                path = out / "structural_predictions" / (digest(r["id"]) + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
    result = {"split": split, "pairs": len(rows), "binding": binary_metrics(
        [r["binding"] if r["binding"] is not None else np.nan for r in predictions], [r["score"] for r in predictions]), "affinity": {}}
    for endpoint in model.cfg.endpoints:
        exact = [r for r in affinity_rows if r["endpoint"] == endpoint and r["lower"] is not None and r["lower"] == r["upper"]]
        result["affinity"][endpoint] = regression_metrics([r["lower"] for r in exact], [r["prediction"] for r in exact])
    result["structural_macro"] = {}
    for head in {r["head"] for r in structural}:
        group = [r for r in structural if r["head"] == head]
        summary = {"complexes": len(group)}
        for metric in ("auroc", "average_precision", "accuracy", "macro_f1", "matched_dice", "matched_iou"):
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                summary[metric] = float(np.mean(values)); summary[metric + "_defined_complexes"] = len(values)
        result["structural_macro"][head] = summary
    write_csv(out / "pair_predictions.csv", predictions)
    write_csv(out / "affinity_predictions.csv", affinity_rows, ["id", "endpoint", "lower", "upper", "prediction"])
    write_json(out / "structural_metrics.json", structural)
    write_json(out / "metrics.json", result)
    write_json(out / "RECEIPT.json", {"checkpoint_sha256": sha256(checkpoint), "dataset_sha256": seal["sha256"],
                                    "selection_seal_sha256": sha256(Path(checkpoint).parent / "SELECTION_SEAL.json"),
                                    "split": split, "provenance": provenance(),
                                    "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    store.close()
    return result
