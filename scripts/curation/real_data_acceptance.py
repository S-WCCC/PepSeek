"""Bounded real-record forward/backward acceptance; never a scientific fit."""
import argparse,json,time
from pathlib import Path
import torch,numpy as np
from pepseek.data import Dataset,collate_chain
from pepseek.features import SequenceFeatures
from pepseek.model import ModelConfig,UnifiedPepSeek
from pepseek.losses import multitask_loss
from pepseek.io import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--esm-checkpoint');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);ds=Dataset(a.dataset);selected={}
    for row in ds.split('train'):
        p=ds.entities['protein'][row['protein_id']];q=ds.entities['peptide'][row['peptide_id']]
        if len(p['sequence'])>600 or len(q['sequence'])>64:continue
        categories=[]
        if row['affinity']:categories.append('quantitative')
        if row['structure_views']:categories.append('structural')
        if row['affinity'] and row['structure_views']:categories.append('both_supervisions')
        if len(p.get('components',[]))>1 and row['structure_views']:categories.append('multi_chain')
        for c in categories:selected.setdefault(c,row)
    if not {'quantitative','structural'}<=set(selected):raise ValueError('Real-data acceptance lacks a required supervision category')
    builder=SequenceFeatures(checkpoint=a.esm_checkpoint,toy_dim=None if a.esm_checkpoint else 32,device='cpu',conformers=0)
    cfg=json.loads((ds.root/'model_config.json').read_text());cfg['esm_dim']=builder.dim
    if not a.esm_checkpoint:cfg.update(hidden=48,pair_dim=24,encoder_layers=1,interaction_layers=2,protein_slots=4,peptide_slots=3,contact_rank=16)
    model=UnifiedPepSeek(ModelConfig(**cfg));results=[];began=time.time()
    for category,row in selected.items():
        inputs={side:collate_chain([builder.build(ds.entities[side][row[side+'_id']],side)],side) for side in ['protein','peptide']}
        label=ds.labels(row);prediction=model(inputs,detail=bool(label),distance='distance_bins' in label)
        loss,terms=multitask_loss(prediction,[row],[label],model.cfg.endpoints);assert torch.isfinite(loss);model.zero_grad();loss.backward()
        grad=sum(float(p.grad.abs().sum()) for p in model.interaction.parameters() if p.grad is not None);assert grad>0
        results.append(dict(category=category,pair_id=row['id'],loss_terms=sorted(terms),finite_loss=True,shared_interaction_gradient_positive=True,protein_components=len(ds.entities['protein'][row['protein_id']]['components'])))
    write_json(out/'RECEIPT.json',dict(status='PASS',records=results,feature_recipe=builder.config,model_config=cfg,optimizer_updates=0,seconds=time.time()-began,meaning='Bounded software/data compatibility acceptance using training records only; not model-performance evidence'))
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
