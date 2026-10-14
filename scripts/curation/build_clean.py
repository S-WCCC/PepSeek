"""Join the two supervision tables into one pair-sampled model dataset.

All observations remain in supervision/. Clean rows contain independent input
identity plus lists of quantitative observations and structural views. Labels are
relative assets; experimental coordinates are never part of forward features.
"""
import argparse,collections,json,math,os,gzip
import numpy as np
from pathlib import Path
from build_catalog import loadjl,jsonl,save,sha,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'clean';out.mkdir(exist_ok=True)
    quant=loadjl(root/'supervision/quantitative.jsonl');struct=loadjl(root/'supervision/structural.jsonl')
    p={r['id']:r for r in loadjl(root/'libraries/protein_components.jsonl')};q={r['id']:r for r in loadjl(root/'libraries/model_peptide_entities.jsonl')}
    assignment={r['observation_id']:r for r in loadjl(root/'splits/observations.jsonl')}
    pc={r['sequence_id']:r['cluster_id'] for r in loadjl(root/'splits/protein_clusters.jsonl')}
    qc={r['sequence_id']:r['cluster_id'] for r in loadjl(root/'splits/peptide_clusters.jsonl')}
    pairs={};proteins={};peptides={};decisions=[];libs={'protein':{},'peptide':{}};observations=[]
    canonical={}
    for k,e in q.items():
        if e.get('smiles'):canonical[k]='QC_'+digest([e['sequence'],e['smiles']])[:24]
    physical_keys={};known_physical=collections.defaultdict(set)
    for r in struct:
        rec=root/'audit/structure_rows'/(r['observation_id']+'.json')
        if not rec.exists():raise RuntimeError('Missing structural disposition '+r['observation_id'])
        st=json.loads(rec.read_text())
        if st['status']!='PASS':continue
        with gzip.open(root/st['map_path'],'rt',encoding='utf-8') as f:mp=json.load(f)
        chains=sorted({x['coordinate_residue_id'].get('auth_asym_id') for x in mp['peptide_map'] if x.get('coordinate_residue_id',{}).get('auth_asym_id')})
        key=digest([r['pdb_id'].upper(),q[r['peptide_id']]['sequence'],chains or [r['source_record_id']]])
        physical_keys[r['observation_id']]=key
        if r['peptide_id'] in canonical:known_physical[key].add(r['peptide_id'])
    for r in struct:
        key=physical_keys.get(r['observation_id'])
        if not key:continue
        if r['peptide_id'] not in canonical:
            known=known_physical.get(key,set());mapped={canonical[k] for k in known}
            if len(mapped)==1:
                r['_identity_entity']=sorted(known)[0];r['_canonical_q']=next(iter(mapped))
            else:r['_canonical_q']='QU_'+key[:24]
    def getrow(r):
        pr=p[r['protein_id']];pe=q[r.get('_identity_entity',r['peptide_id'])];part=assignment[r['observation_id']];cid=part['component_id']
        po={k:pr[k] for k in ['id','sequence','components','component_ids','component_lengths']};po.update(cluster_id=cid,protein_homology_cluster_ids=[pc[i] for i in pr['component_ids']])
        qe={k:v for k,v in pe.items() if k not in ['reason','mapping_status']};qe.update(cluster_id=cid,peptide_similarity_cluster_id=qc[pe['peptide_sequence_id']])
        qe['id']=r.get('_canonical_q',canonical.get(pe['id'],pe['id']));qe['chemical_identity_id']=qe['id']
        if pr['id'] in proteins and proteins[pr['id']]['cluster_id']!=cid:raise AssertionError('Protein component split conflict')
        if qe['id'] in peptides and peptides[qe['id']]['cluster_id']!=cid:raise AssertionError('Peptide entity split conflict')
        proteins[po['id']]=po;peptides[qe['id']]=qe
        pid='CPAIR_'+digest([pr['id'],qe['id']])[:24]
        row=pairs.setdefault(pid,dict(id=pid,protein_id=pr['id'],peptide_id=qe['id'],split=part['split'],component_id=cid,affinity=[],structure_views=[],observation_ids=[],source_pair_ids=[]))
        if row['split']!=part['split']:raise AssertionError('Pair split leakage')
        row['observation_ids'].append(r['observation_id'])
        if r['pair_id'] not in row['source_pair_ids']:row['source_pair_ids'].append(r['pair_id'])
        return row
    for r in quant:
        pe=q[r['peptide_id']]
        if pe['mapping_status']!='EXPLICIT_CHEMISTRY_MAPPED':
            decisions.append(dict(observation_id=r['observation_id'],status='HOLD',reason='CHEMICAL_ATOM_RESIDUE_MAPPING'));continue
        m=r['measurement'];row=getrow(r)
        lo=m.get('upper_M');hi=m.get('lower_M')
        lower=-math.log10(lo) if lo and lo>0 else None;upper=-math.log10(hi) if hi and hi>0 else None
        ep={'Kd':'pKd','Ki':'pKi','IC50':'pIC50_assay_dependent','EC50':'pEC50','Km':'pKm'}[m['endpoint']]
        row['affinity'].append(dict(observation_id=r['observation_id'],endpoint=ep,lower=lower,upper=upper,relation=m['p_relation'],assay=m['assay'],assay_conditions=m['assay_conditions'],reliable_ranking=False,group_id=None,approximate=m['relation']=='~'))
        observations.append(dict(r,split=row['split'],clean_pair_id=row['id']))
        decisions.append(dict(observation_id=r['observation_id'],status='USE',reason='VERIFIED_MEASUREMENT_AND_CHEMISTRY',split=row['split']))
    structural_obs=[];alias={};view_keys=collections.defaultdict(dict);empty_structure_components={};checked_structure_components=set()
    for r in struct:
        rec=root/'audit/structure_rows'/(r['observation_id']+'.json')
        if not rec.exists():raise RuntimeError('Missing structural disposition '+r['observation_id'])
        st=json.loads(rec.read_text())
        if st['status']!='PASS':
            decisions.append(dict(observation_id=r['observation_id'],status='HOLD',reason=st['reason']));continue
        row=getrow(r)
        # Same pair + identical supervised arrays are aliases of one training view.
        # Distinct sources stay visible in the two supervision tables.
        key=st['label_sha256'];prior=view_keys[row['id']].get(key)
        if prior:
            prior['observation_ids'].append(r['observation_id']);prior['map_paths'].append('../'+st['map_path']);alias[r['observation_id']]=prior['observation_ids'][0]
        else:
            rel='../'+st['label_path'];v=dict(path=rel,label_sha256=key,observation_ids=[r['observation_id']],map_paths=['../'+st['map_path']]);view_keys[row['id']][key]=v;row['structure_views'].append(v)
        for e in st['structure_libraries']:
            if e['id'] not in checked_structure_components:
                with np.load(root/e['path'],allow_pickle=False) as z:
                    if not len(z['atom_residue']):empty_structure_components[e['id']]=dict(e,reason='NO_OBSERVED_ATOMS_COMPONENT_REMAINS_MASKED_IN_COMPLEX')
                checked_structure_components.add(e['id'])
            if e['id'] in empty_structure_components:continue
            l=libs[e['side']].setdefault(e['id'],dict(e,observation_ids=[],map_paths=[]));l['observation_ids'].append(r['observation_id']);l['map_paths'].append(e['map_path']);l.pop('observation_id',None);l.pop('map_path',None)
        structural_obs.append(dict(r,split=row['split'],clean_pair_id=row['id'],labels={k:st[k] for k in ['label_path','label_sha256','map_path','map_sha256','positive_contacts','negative_contacts','masked_contacts','distance_labels']},alias_of=alias.get(r['observation_id'])))
        decisions.append(dict(observation_id=r['observation_id'],status='USE',reason='MAPPED_HEAVY_ATOM_CONTACT_MASK_AUDIT',split=row['split']))
    # relative_file intentionally prevents traversal. Keep label pointers rooted in
    # the dataset manifest directory through a dedicated in-tree linked label set.
    (out/'labels').mkdir(exist_ok=True)
    for row in pairs.values():
        for v in row['structure_views']:
            src=(out/v['path']).resolve();dst=out/'labels'/src.name
            if not dst.exists():
                try:os.link(src,dst)
                except OSError:
                    import shutil;shutil.copy2(src,dst)
            v['path']='labels/'+src.name
        row['contrastive_positive']=bool(row['structure_views'])
        row['contrastive_evidence']='observed_mapped_complex_contact;not_experimental_binary_assay' if row['structure_views'] else None
        # No measured binary binding label is invented; retrieval uses its own field.
        row['sampling_policy']='one_input_pair_per_epoch;rotate_structural_views;mean_quantitative_observations'
    jsonl(out/'proteins.jsonl',proteins.values());jsonl(out/'peptides.jsonl',peptides.values());jsonl(out/'pairs.jsonl',pairs.values())
    save(out/'dataset.json',dict(schema_version=1,proteins='proteins.jsonl',peptides='peptides.jsonl',pairs='pairs.jsonl',split_policy='cluster_disjoint',cluster_method='MMseqs2 protein30% identity/80% shorter; BLASTP-short peptide50% identity/80% both; connected components include PDB, publications and verified links; seed17',release='curated_dual_v1_20261013',unit='input pair with two independently masked supervision lists',label_policy='heavy-atom contact <=8A; negatives require complete heavy atoms on both residues; missing NaN; distance bins[4.5,8,12,16]',historical_test_exposure=True))
    endpoints=['pKd','pKi','pIC50_assay_dependent','pEC50','pKm']
    model=dict(esm_dim=1280,hidden=384,pair_dim=128,encoder_layers=2,interaction_layers=6,protein_slots=16,peptide_slots=8,contact_rank=64,dropout=.1,distance_bins=5,endpoints=endpoints)
    save(out/'model_config.json',model)
    save(out/'training_config_example.json',dict(epochs=50,max_steps=100000,batch_size=2,patience=10,lr=1e-4,weight_decay=1e-4,gradient_clip=1.,seed=17,cpu_threads=4,pair_chunk=4,loss_weights=dict(affinity=1.,contact=1.,interface=1.,distance=.2,contrastive=.2,ranking=0.,region=0.,binding=0.),scientific_run_requires_separate_user_run_plan=True))
    jsonl(root/'supervision/quantitative_clean.jsonl',observations);jsonl(root/'supervision/structural_clean.jsonl',structural_obs);jsonl(root/'audit/clean_dispositions.jsonl',decisions)
    for side in libs:jsonl(root/f'libraries/{side}_structure_index.jsonl',libs[side].values())
    jsonl(root/'audit/empty_structure_components.jsonl',empty_structure_components.values())
    usable_seq={side:{} for side in ['protein','peptide']}
    for e in proteins.values():
        for seq in e['components']:usable_seq['protein'][digest(seq)]=seq
    for e in peptides.values():usable_seq['peptide'][digest(e['sequence'])]=e['sequence']
    for side in usable_seq:
        records=[dict(id=side[0].upper()+'_'+h[:24],sequence=s,length=len(s)) for h,s in sorted(usable_seq[side].items())]
        jsonl(root/f'libraries/{side}_sequences_clean.jsonl',records)
        with (root/f'libraries/{side}_sequences_clean.fasta').open('w') as f:
            for e in records:f.write('>'+e['id']+'\n'+e['sequence']+'\n')
    splitstats={}
    for split in ['train','valid','test']:
        pr=[r for r in pairs.values() if r['split']==split]
        splitstats[split]=dict(input_pairs=len(pr),quantitative_observations=sum(len(r['affinity']) for r in pr),structural_observations=sum(x['split']==split for x in structural_obs),distinct_structure_views=sum(len(r['structure_views']) for r in pr),protein_inputs=len({r['protein_id'] for r in pr}),peptide_inputs=len({r['peptide_id'] for r in pr}),pairs_with_both_supervisions=sum(bool(r['affinity']) and bool(r['structure_views']) for r in pr),components=len({r['component_id'] for r in pr}),endpoints=dict(collections.Counter(x['endpoint'] for r in pr for x in r['affinity'])))
    result=dict(status='CLEAN_BUILT_VALIDATION_PENDING',quantitative_observations=len(observations),structural_observations=len(structural_obs),input_pairs=len(pairs),split_counts=splitstats,unique_protein_sequences_clean=len(usable_seq['protein']),unique_peptide_sequences_clean=len(usable_seq['peptide']),unique_protein_structures=len(libs['protein']),unique_peptide_structures=len(libs['peptide']),duplicate_structural_views=len(alias),clean_holds=dict(collections.Counter(x['reason'] for x in decisions if x['status']!='USE')),no_experimental_negatives_invented=True,coordinates_not_model_inputs=True)
    result['empty_structure_components_not_counted']=len(empty_structure_components)
    save(root/'audit/CLEAN_RECEIPT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
