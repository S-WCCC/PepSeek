import numpy as np
import torch
from pepseek.features import SequenceFeatures,FeatureStore
from pepseek.data import collate_chain,Dataset
from pepseek.model import ModelConfig,UnifiedPepSeek
from pepseek.losses import multitask_loss
from pepseek.demo import create_demo
from pepseek.io import jsonl,write_jsonl

def test_unknown_chemistry_is_masked_and_can_train():
    b=SequenceFeatures(toy_dim=16,conformers=2)
    q=b.build({'id':'q','sequence':'ACXDE','chemistry_mode':'unknown'},'peptide')
    assert q['atom_features'].shape==(0,12) and not q['geometry_mask'].any()
    p=b.build({'id':'p','sequence':'ACDEFGHIK'},'protein')
    m=UnifiedPepSeek(ModelConfig(esm_dim=16,hidden=24,pair_dim=12,encoder_layers=1,interaction_layers=1,protein_slots=2,peptide_slots=2,contact_rank=8))
    o=m({'protein':collate_chain([p],'protein'),'peptide':collate_chain([q],'peptide')},True,True)
    o['contact_logits'].square().mean().backward()
    assert torch.isfinite(o['score']).all()
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in m.interaction.parameters())

def test_component_boundaries_no_false_bond_and_reset_positions(tmp_path):
    b=SequenceFeatures(toy_dim=16)
    e={'id':'p','sequence':'ACDEFGH','components':['ACD','EFGH']}
    f=b.build(e,'protein');edges=set(map(tuple,f['edges'].T.tolist()))
    assert (2,3) not in edges and (3,2) not in edges
    assert f['position_features'][0,0]==0 and f['position_features'][3,0]==0
    s=FeatureStore(tmp_path/'f.db',b.config)
    assert s.key(e,'protein')!=s.key(dict(e,components=['ACDEFGH']),'protein')
    s.close()

def test_repeated_measurements_do_not_multiply_pair_loss():
    a={'endpoint':'pKd','lower':1.,'upper':1.}
    o={'score':torch.zeros(2,requires_grad=True),'affinity':torch.tensor([[2.],[4.]],requires_grad=True)}
    one=multitask_loss(o,[{'affinity':[a]},{'affinity':[a]}],[{},{}],['pKd'])[0]
    repeated=multitask_loss(o,[{'affinity':[a]*10},{'affinity':[a]}],[{},{}],['pKd'])[0]
    torch.testing.assert_close(one,repeated)

def test_structure_views_rotate_one_pair_once(tmp_path):
    ds=Dataset(create_demo(tmp_path/'ds'))
    rows=jsonl(ds.root/'pairs.jsonl');r=next(x for x in rows if x['split']=='train')
    np.savez(ds.root/'v1.npz',contact=np.ones((len(ds.entities['protein'][r['protein_id']]['sequence']),len(ds.entities['peptide'][r['peptide_id']]['sequence']))))
    np.savez(ds.root/'v2.npz',contact=np.zeros((len(ds.entities['protein'][r['protein_id']]['sequence']),len(ds.entities['peptide'][r['peptide_id']]['sequence']))))
    r['structure_views']=[{'path':'v1.npz'},{'path':'v2.npz'}];r.pop('structure_labels',None)
    write_jsonl(ds.root/'pairs.jsonl',rows);ds=Dataset(ds.manifest)
    a=ds.training_rows(0);b=ds.training_rows(1)
    assert len(a)==len(ds.split('train')) and a[0]['structure_labels']!=b[0]['structure_labels']
    assert {'v1.npz','v2.npz'}<=set(ds.seal()['files'])
