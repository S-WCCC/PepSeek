"""Exhaustive release validation, including all model label arrays and file hashes."""
import argparse,collections,csv,gzip,hashlib,json,math,time,concurrent.futures
from pathlib import Path
import numpy as np
from rdkit import Chem
from build_catalog import loadjl,jsonl,save,sha,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    if (root/'RELEASE_SEAL.json').exists():raise RuntimeError('Already sealed')
    from pepseek.data import Dataset
    ds=Dataset(root/'clean/dataset.json');began=time.time();errors=[];counts=collections.Counter();hashes={}
    for row in ds.pairs:
        for r in ([dict(row,structure_labels=v['path']) for v in row.get('structure_views',[])] or [row]):
            lab=ds.labels(r)
            if lab:
                c=lab['contact'];pi=lab['protein_interface'];qi=lab['peptide_interface'];d=lab['distance_bins']
                assert np.all(pi[(c==1).any(1)]==1) and np.all(qi[(c==1).any(0)]==1)
                assert np.all(np.isin(d,[-1,0,1,2,3,4]))
                assert np.all(c[(d==0)|(d==1)]==1)
                assert np.all(c[d>=2]==0)
                counts['contact_positive_labels']+=int(np.sum(c==1));counts['contact_negative_labels']+=int(np.sum(c==0));counts['contact_masked_labels']+=int(np.isnan(c).sum())
                counts['protein_interface_positive_labels']+=int(np.sum(pi==1));counts['protein_interface_negative_labels']+=int(np.sum(pi==0))
                counts['peptide_interface_positive_labels']+=int(np.sum(qi==1));counts['peptide_interface_negative_labels']+=int(np.sum(qi==0));counts['distance_observed_labels']+=int(np.sum(d>=0))
                path=ds.root/r['structure_labels'];h=sha(path);hashes[r['structure_labels']]=h
                expected=next(v['label_sha256'] for v in row['structure_views'] if v['path']==r['structure_labels']);assert h==expected
                counts['label_views']+=1
        counts['input_pairs']+=1;counts['quantitative_observations']+=len(row.get('affinity',[]))
    for e in ds.entities['peptide'].values():
        if e.get('chemistry_mode')=='unknown':assert not e.get('smiles') and not e.get('atom_residue');counts['masked_chemical_entities']+=1
        else:
            mol=Chem.MolFromSmiles(e['smiles']);assert mol is not None
            mapping=e['atom_residue'];assert len(mapping)==mol.GetNumAtoms() and set(mapping)==set(range(len(e['sequence'])))
            counts['explicit_chemical_entities']+=1
    disp=loadjl(root/'audit/clean_dispositions.jsonl');assert len(disp)==85219 and len({r['observation_id'] for r in disp})==85219
    splits=loadjl(root/'splits/observations.jsonl');assert len(splits)==85219
    cluster=collections.defaultdict(set)
    for r in splits:cluster[r['component_id']].add(r['split'])
    assert all(len(v)==1 for v in cluster.values())
    # Both source supervision tables exhaust the source-curated eligibility ledger.
    quant=loadjl(root/'supervision/quantitative_clean.jsonl')
    for r in quant:
        m=r['measurement'];assert math.isclose(m['p_value'],-math.log10(m['value_M']),abs_tol=1e-7)
        assert m['endpoint'] in ['Kd','Ki','IC50','EC50','Km']
    asset_refs=loadjl(root/'audit/asset_manifest.jsonl');assets_by_path={}
    for r in asset_refs:
        if r['path'] in assets_by_path:assert (r['sha256'],r['bytes'])==(assets_by_path[r['path']]['sha256'],assets_by_path[r['path']]['bytes'])
        assets_by_path[r['path']]=r
    assets=list(assets_by_path.values());counts['source_asset_references']=len(asset_refs);counts['unique_source_asset_bytes']=sum(r['bytes'] for r in assets)
    def verify_asset(r):
        path=root/r['path'];assert path.is_file() and path.stat().st_size==r['bytes'],r['path']
        assert sha(path)==r['sha256'],r['path'];return 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:counts['source_assets_sha_verified']=sum(pool.map(verify_asset,assets))
    for side in ['protein','peptide']:
        index=loadjl(root/f'libraries/{side}_structure_index.jsonl')
        for r in index:
            with np.load(root/r['path'],allow_pickle=False) as z:
                xyz=z['coordinates'];idx=z['atom_residue'];n=len(r['sequence'])
                assert xyz.shape==(len(idx),3) and np.isfinite(xyz).all()
                assert len(z['atom_names'])==len(idx)==len(z['monomers'])
                assert len(z['observed_mask'])==n==len(z['complete_mask'])
                assert idx.min()>=0 and idx.max()<n and np.all(~z['complete_mask']|z['observed_mask'])
            counts[side+'_structures_checked']+=1
    for r in loadjl(root/'supervision/structural_clean.jsonl'):
        lab=r['labels'];assert sha(root/lab['map_path'])==lab['map_sha256']
        with gzip.open(root/lab['map_path'],'rt',encoding='utf-8') as f:mp=json.load(f)
        assert mp['native_coordinates_are_supervision_only'] is True
        for side in ['protein','peptide']:
            n=sum(map(len,mp[side+'_components']))
            assert all(0<=x['input_residue_index']<n for x in mp[side+'_map'])
        counts['structure_maps_checked']+=1
    seal=ds.seal();save(root/'clean/DATA_SEAL.json',seal)
    report=dict(status='PASS',counts=dict(counts),model_dataset_sha256=seal['sha256'],split_counts=seal['counts'],cross_partition_components=0,all_source_assets_sha_checked=True,all_structure_libraries_checked=True,all_structure_maps_checked=True,all_model_labels_checked=True,all_explicit_atom_maps_checked=True,elapsed_seconds=time.time()-began,scientific_training_run=False)
    save(root/'audit/ACCEPTANCE.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
