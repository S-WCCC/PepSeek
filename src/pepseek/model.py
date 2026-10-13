"""One pool--interact--unpool backbone for scoring and structural supervision.

No native complex coordinates or labels are accepted by forward(). Protein and
peptide preprocessing is independent. Only generated, intrapeptide distances
enter the geometry encoder; no cross-molecule distance is an input.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    esm_dim: int = 1280
    hidden: int = 384
    pair_dim: int = 128
    encoder_layers: int = 2
    interaction_layers: int = 6
    protein_slots: int = 16
    peptide_slots: int = 8
    contact_rank: int = 64
    dropout: float = 0.1
    distance_bins: int = 5
    endpoints: tuple = ("pKd", "pKi", "pIC50_binding", "pIC50_functional", "pEC50")

    def __post_init__(self):
        self.endpoints = tuple(self.endpoints)
        for name in ("esm_dim", "hidden", "pair_dim", "encoder_layers", "interaction_layers",
                     "protein_slots", "peptide_slots", "contact_rank", "distance_bins"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1 or not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("Invalid dropout or endpoint names")


def masked_mean(x, mask, dim):
    return (x * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).clamp_min(1).unsqueeze(-1)


class SparseGraphBlock(nn.Module):
    """Sequence/bond graph messages, O(edges); no dense L x L allocation."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.message = nn.Linear(dim, dim, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x, edges, edge_weight=None):
        z = self.norm(x)
        acc = torch.zeros_like(z)
        degree = z.new_zeros((len(z), 1))
        if edges.numel():
            src, dst = edges
            weights = z.new_ones((len(src), 1)) if edge_weight is None else edge_weight[:, None].to(z)
            acc.index_add_(0, dst, self.message(z[src]) * weights)
            degree.index_add_(0, dst, weights)
        return x + self.update(torch.cat([z, acc / degree.clamp_min(1)], -1))


class ChainEncoder(nn.Module):
    def __init__(self, cfg, chemical=False):
        super().__init__()
        d = cfg.hidden
        self.esm = nn.Sequential(nn.LayerNorm(cfg.esm_dim), nn.Linear(cfg.esm_dim, d))
        self.aa = nn.Embedding(21, d)
        self.position = nn.Linear(3, d, bias=False)
        self.layers = nn.ModuleList([SparseGraphBlock(d, cfg.dropout) for _ in range(cfg.encoder_layers)])
        self.chemical = chemical
        if chemical:
            self.atom_input = nn.Linear(12, d)
            self.atom_layers = nn.ModuleList([SparseGraphBlock(d, cfg.dropout) for _ in range(2)])
            self.geometry_filter = nn.Sequential(nn.Linear(16, d), nn.SiLU(), nn.Linear(d, d))
            self.geometry_value = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch):
        mask = batch["mask"]
        if mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError("Each chain must have at least one unpadded residue")
        esm = batch["esm"]
        if esm.shape[:2] != mask.shape or not torch.isfinite(esm).all():
            raise ValueError("Invalid residue ESM features")
        b, length = mask.shape
        idx = torch.arange(length, device=mask.device)[None].expand(b, -1)
        n = mask.sum(1, keepdim=True)
        pos = torch.stack([idx / (n - 1).clamp_min(1), torch.exp(-idx / 4),
                           torch.exp(-(n - 1 - idx).clamp_min(0) / 4)], -1)
        x = self.esm(esm) + self.aa(batch["aa"]) + self.position(pos)
        edges = batch["edges"]
        if self.chemical:
            atoms = self.atom_input(batch["atom_features"])
            for layer in self.atom_layers:
                atoms = layer(atoms, batch["atom_edges"], batch["atom_edge_weight"])
            residue = batch["atom_residue"]  # global padded residue index
            total = atoms.new_zeros((b * length, atoms.shape[-1]))
            count = atoms.new_zeros((b * length, 1))
            total.index_add_(0, residue, atoms)
            count.index_add_(0, residue, atoms.new_ones((len(atoms), 1)))
            x = x + (total / count.clamp_min(1)).reshape_as(x)
        x = x * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x.reshape(b * length, -1), edges).reshape(b, length, -1)
            x = x * mask.unsqueeze(-1)
        if not self.chemical:
            return {"h": self.norm(x)[:, None] * mask[:, None, :, None],
                    "mask": mask, "conformer_mask": mask.new_ones((b, 1))}
        coords, gm, cm = batch["conformer_coords"], batch["geometry_mask"], batch["conformer_mask"]
        if not cm.any(1).all():
            raise ValueError("No usable conformer/sequence fallback for a peptide")
        # Pairwise distances are within ONE peptide. Translation/rotation invariant.
        c = coords.shape[1]
        base = x[:, None].expand(-1, c, -1, -1)
        geom_chunks = []
        centers = torch.linspace(0, 24, 16, device=x.device, dtype=x.dtype)
        for start in range(0, length, 32):
            dist = torch.cdist(coords[:, :, start:start + 32].float(), coords.float()).to(x.dtype)
            valid = gm[:, :, start:start + 32, None] & gm[:, :, None, :] & cm[:, :, None, None]
            self_edge = torch.arange(start, min(start + 32, length), device=x.device)[:, None] == torch.arange(length, device=x.device)[None]
            valid = valid & ~self_edge & (dist <= 24)
            rbf = torch.exp(-((dist[..., None] - centers) / 1.6).square())
            # Scalar edge gates plus shared value messages; geometry has no frame dependence.
            gates = torch.sigmoid(self.geometry_filter(rbf)) * valid[..., None]
            vals = self.geometry_value(base)[:, :, None]
            geom_chunks.append((gates * vals).sum(3) / valid.sum(3).clamp_min(1)[..., None])
        h = self.norm(base + torch.cat(geom_chunks, 2))
        return {"h": h * mask[:, None, :, None], "mask": mask, "conformer_mask": cm}


class SoftPool(nn.Module):
    def __init__(self, dim, slots):
        super().__init__()
        self.assignment = nn.Linear(dim, slots)

    def forward(self, x, mask):
        assignment = self.assignment(x).softmax(-1) * mask[..., None]
        mass = assignment.sum(1).clamp_min(1e-6)
        nodes = torch.einsum("blk,bld->bkd", assignment, x) / mass[..., None]
        return nodes, assignment


class InteractionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, z = cfg.hidden, cfg.pair_dim
        self.protein_pool = SoftPool(d, cfg.protein_slots)
        self.peptide_pool = SoftPool(d, cfg.peptide_slots)
        self.pair = nn.Sequential(nn.Linear(4 * d, z), nn.GELU(), nn.LayerNorm(z),
                                  nn.Linear(z, z), nn.GELU())
        self.attention = nn.Linear(z, 1)
        self.to_protein = nn.Linear(z, d)
        self.to_peptide = nn.Linear(z, d)
        self.p_update = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.q_update = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout))

    def forward(self, p, q, pm, qm):
        pn, ps = self.protein_pool(p, pm)
        qn, qs = self.peptide_pool(q, qm)
        pp = pn[:, :, None].expand(-1, -1, qn.shape[1], -1)
        qq = qn[:, None].expand(-1, pn.shape[1], -1, -1)
        z = self.pair(torch.cat([pp, qq, pp * qq, (pp - qq).abs()], -1))
        logits = self.attention(z).squeeze(-1)
        p_message = (logits.softmax(2)[..., None] * self.to_protein(z)).sum(2)
        q_message = (logits.softmax(1)[..., None] * self.to_peptide(z)).sum(1)
        # The candidate-conditioned messages RETURN to residues at every layer.
        p = (p + self.p_update(torch.bmm(ps, p_message))) * pm[..., None]
        q = (q + self.q_update(torch.bmm(qs, q_message))) * qm[..., None]
        return p, q, z, logits


def mixture_logit(logits, weights):
    probability = (logits.sigmoid() * weights).sum(1).clamp(1e-6, 1 - 1e-6)
    return torch.logit(probability)


class UnifiedPepSeek(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg
        self.protein_encoder = ChainEncoder(cfg)
        self.peptide_encoder = ChainEncoder(cfg, chemical=True)
        self.interaction = nn.ModuleList([InteractionBlock(cfg) for _ in range(cfg.interaction_layers)])
        d, z, rank = cfg.hidden, cfg.pair_dim, cfg.contact_rank
        self.joint = nn.Sequential(nn.Linear(z + 2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.score = nn.Linear(d, 1)
        self.affinity = nn.Linear(d, len(cfg.endpoints))
        self.conformer_gate = nn.Linear(d, 1)
        self.p_site = nn.Linear(d, 1)
        self.q_site = nn.Linear(d, 1)
        self.p_contact = nn.Linear(d, rank)
        self.q_contact = nn.Linear(d, rank)
        self.contact_bias = nn.Parameter(torch.zeros(()))
        self.geometry = nn.Linear(rank, cfg.distance_bins)
        self.region_membership = nn.Linear(d, cfg.protein_slots)

    def encode(self, chain, side):
        if side not in ("protein", "peptide"):
            raise ValueError("side must be protein or peptide")
        return getattr(self, side + "_encoder")(chain)

    def forward(self, inputs, detail=True, distance=False):
        if set(inputs) != {"protein", "peptide"}:
            raise ValueError("Only independent protein/peptide inputs are accepted; labels are separate")
        return self.interact(self.encode(inputs["protein"], "protein"),
                             self.encode(inputs["peptide"], "peptide"), detail, distance)

    def interact(self, protein, peptide, detail=True, distance=False):
        """Cached *preinteraction* features are reusable only with frozen weights."""
        b, c, lq, d = peptide["h"].shape
        if protein["h"].shape[0] != b:
            raise ValueError("Batch sizes must match (expand query explicitly)")
        lp = protein["h"].shape[2]
        p = protein["h"].expand(-1, c, -1, -1).reshape(b * c, lp, d)
        q = peptide["h"].reshape(b * c, lq, d)
        pm = protein["mask"][:, None].expand(-1, c, -1).reshape(b * c, lp)
        qm = peptide["mask"][:, None].expand(-1, c, -1).reshape(b * c, lq)
        for block in self.interaction:
            p, q, z, logits = block(p, q, pm, qm)
        attention = logits.flatten(1).softmax(-1).reshape_as(logits)
        interaction_summary = (z * attention[..., None]).sum((1, 2))
        joint = self.joint(torch.cat([interaction_summary, masked_mean(p, pm, 1), masked_mean(q, qm, 1)], -1)).reshape(b, c, d)
        weights = self.conformer_gate(joint).squeeze(-1).masked_fill(~peptide["conformer_mask"], -torch.inf).softmax(1)
        out = {"score": (self.score(joint).squeeze(-1) * weights).sum(1),
               "affinity": (self.affinity(joint) * weights[..., None]).sum(1),
               "conformer_weights": weights}
        if not detail:
            return out
        p, q = p.reshape(b, c, lp, d), q.reshape(b, c, lq, d)
        out["protein_interface_logits"] = mixture_logit(self.p_site(p).squeeze(-1), weights[..., None])
        out["peptide_interface_logits"] = mixture_logit(self.q_site(q).squeeze(-1), weights[..., None])
        # Region slots are latent. Their residue memberships need explicit region supervision.
        out["region_membership_logits"] = mixture_logit(self.region_membership(p).permute(0, 1, 3, 2), weights[..., None, None])
        # Interaction slots and region decoder slots have independent indexing.
        out["interaction_slot_attention"] = (attention.sum(2).reshape(b, c, -1) * weights[..., None]).sum(1)
        ph, qh = self.p_contact(p), self.q_contact(q)
        contact, geometry = [], []
        for start in range(0, lp, 32):
            product = ph[:, :, start:start + 32, None] * qh[:, :, None]
            raw = product.sum(-1) / self.cfg.contact_rank ** 0.5 + self.contact_bias
            contact.append(mixture_logit(raw, weights[..., None, None]))
            if distance:
                probs = (self.geometry(product).softmax(-1) * weights[..., None, None, None]).sum(1)
                geometry.append(probs.clamp_min(1e-8).log())
        out["contact_logits"] = torch.cat(contact, 1)
        if distance:
            out["distance_log_probs"] = torch.cat(geometry, 1)
        out["pair_mask"] = protein["mask"][:, :, None] & peptide["mask"][:, None, :]
        out["protein_mask"], out["peptide_mask"] = protein["mask"], peptide["mask"]
        return out

    def configuration(self):
        return asdict(self.cfg)


def select_encoded(encoded, indices):
    return {key: value[indices] for key, value in encoded.items()}


def score_matrix(model, proteins, peptides, chunk_size=16):
    """Both-direction scores from THE interaction backbone, not a dot-product bypass."""
    np_, nq = proteins["h"].shape[0], peptides["h"].shape[0]
    chunks = []
    for start in range(0, np_ * nq, chunk_size):
        ids = torch.arange(start, min(start + chunk_size, np_ * nq), device=proteins["h"].device)
        chunks.append(model.interact(select_encoded(proteins, ids // nq), select_encoded(peptides, ids % nq), detail=False)["score"])
    return torch.cat(chunks).reshape(np_, nq)
