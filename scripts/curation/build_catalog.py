"""Normalize sealed curation into two observation tables; never edit source releases.

All paths in distributable tables are relative. Local acquisition paths are kept
only in the private staging manifest. A record is an observation, not a unique pair.
"""
import argparse, collections, csv, gzip, hashlib, json, shutil
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def loadjl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(l) for l in f]
def cleanseq(s):
    s=''.join(s.split()).upper()
    if not s or set(s)-set('ACDEFGHIKLMNPQRSTVWYXUOBZJ'):raise ValueError('Invalid sequence '+repr(s[:80]))
    return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--curated',required=True);ap.add_argument('--integrated',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    cur,base,out=map(Path,(a.curated,a.integrated,a.out))
    if (out/'RELEASE_SEAL.json').exists():raise RuntimeError('Sealed output is immutable')
    for x in ('raw/snapshots','libraries','supervision','audit','staging','splits','labels','maps','coordinates','clean'):(out/x).mkdir(parents=True,exist_ok=True)
    # Preserve the complete original normalized ledger, including exclusions.
    for name in ('RELEASE_SEAL.json','SUMMARY.json','POLICY.json','ALL_DATA_OVERVIEW.json'):
        shutil.copy2(cur/name,out/'raw/snapshots'/name)
    shutil.copy2(cur/'data/records.jsonl.gz',out/'raw/snapshots/ppikb_records.jsonl.gz')
    with gzip.open(cur/'data/records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(l) for l in f]
    pp={r['record_id']:r for r in records}
    expected={str(base/r['path']):r['sha256'] for r in rows(base/'audit/coordinate_manifest.csv') if r['path'] and r['sha256']}
    expected.update({str(base/r['path']):r['sha256'] for r in rows(base/'audit/existing_asset_files.csv') if r['path'] and r['sha256']})
    old={r['source_record_id']:r for r in rows(base/'data/integrated_structure_source_registry.csv')}
    links=list(rows(cur/'data/confirmed_joint_links.csv'))
    bys=collections.defaultdict(list)
    for l in links:bys[l['structure_id']].append(l)
    prots={};peps={};complexes={};chemicals={};pairs={};assets={};quant=[];structures=[];exceptions=[]
    def protein(comps):
        cs=[cleanseq(s) for s in comps];ids=[]
        for s in cs:
            i='P_'+digest(s)[:24];prots.setdefault(i,dict(id=i,sequence=s,length=len(s),sequence_sha256=hashlib.sha256(s.encode()).hexdigest()));ids.append(i)
        # Component order is retained to make label tensor indexing unambiguous.
        i='PC_'+digest(ids)[:24];complexes.setdefault(i,dict(id=i,component_ids=ids,components=cs,sequence=''.join(cs),component_lengths=list(map(len,cs))))
        return i
    def peptide(s,chemical,scope):
        s=cleanseq(s);i='Q_'+digest(s)[:24];peps.setdefault(i,dict(id=i,sequence=s,length=len(s),sequence_sha256=hashlib.sha256(s.encode()).hexdigest()))
        k='QC_'+digest([i,chemical or {'unresolved_identity_scope':scope}])[:24]
        chemicals.setdefault(k,dict(id=k,peptide_sequence_id=i,sequence=s,chemical_identity=chemical,chemistry_mode='explicit' if chemical else 'unknown',identity_scope=scope))
        return k
    def pair(p,q):
        i='PAIR_'+digest([p,q])[:24];pairs.setdefault(i,dict(pair_id=i,protein_id=p,peptide_id=q,observation_ids=[]));return i
    def asset(path,kind='coordinates'):
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(path)
        key=str(path)
        if key not in assets:
            h=expected.get(key) or sha(path);suffix=''.join(path.suffixes);rel=kind+'/'+h[:2]+'/'+h+suffix
            assets[key]=dict(source_path=key,path=rel,sha256=h,bytes=path.stat().st_size,hash_source='SEALED_PARENT_MANIFEST' if key in expected else 'RECOMPUTED_LOCAL')
        return {k:v for k,v in assets[key].items() if k!='source_path'}
    for r in records:
        if r['record_kind']!='measurement' or not r['supervision']['quantitative']:continue
        p=protein([x['sequence'] for x in r['protein']['components']]);q=peptide(r['peptide']['sequence'],dict(smiles=r['peptide']['canonical_isomeric_smiles'],modifications=r['peptide']['modifications']), 'verified_measurement_identity')
        pid=pair(p,q);oid='M_'+r['record_id'];pairs[pid]['observation_ids'].append(oid)
        quant.append(dict(observation_id=oid,pair_id=pid,protein_id=p,peptide_id=q,source='PPIKB',source_record_id=r['record_id'],measurement=r['measurement'],provenance=r['provenance'],identity_evidence=r['decision'],verified_joint_links=r['joint_links'],reported_protein_role=r['protein']['sequence_role']))
    roots=json.loads((cur/'ASSET_ROOTS.json').read_text())
    for r in rows(cur/'data/combined_structure_inventory.csv'):
        rid=r['record_id'];source=r['source'];rec=pp.get(rid);prior=old.get(rid)
        comps=json.loads(r['protein_components_json']);p=protein(comps)
        q=peptide(r['peptide_sequence'],None,[source,rid])
        # Identical sequences alone do not establish the same chemical entity.
        # Reuse a quantitative identity only for an audited exact-coordinate link.
        joint=bys.get(rid,[])
        exact=[]
        for lk in joint:
            m=next((m for m in quant if m['source_record_id']==lk['measurement_id']),None)
            if m and complexes[p]['components']==complexes[m['protein_id']]['components'] and cleanseq(r['peptide_sequence'])==chemicals[m['peptide_id']]['sequence']:
                exact.append(m['peptide_id'])
        if len(set(exact))==1:q=exact[0]
        pid=pair(p,q);oid='S_'+digest([source,rid])[:24];pairs[pid]['observation_ids'].append(oid)
        if rec:
            st=rec['structure'];c=st['coordinate'];coords=[asset(Path(roots[c['root']])/c['path'])]
            if coords[0]['sha256']!=c['sha256']:raise RuntimeError('Coordinate checksum mismatch '+rid)
            labs=[asset(Path(roots[x['root']])/x['path'],'inherited_labels') for x in st['labels']]
            maps=[asset(Path(roots[x['root']])/x['path'],'inherited_maps') for x in st['maps']]
            pc=st['protein_label_asym_ids'];qc=st['peptide_label_asym_ids'];oldkey=None
            provenance=rec['provenance'];context=st['coordinate_context']
        else:
            paths=json.loads(prior['coordinate_reference'])
            coords=[dict(asset(base/x),role='receptor' if 'receptor.pdb' in x.lower() else 'peptide' if 'peptide.pdb' in x.lower() else 'complex') for x in paths]
            labs=[];maps=[];pc=prior['protein_chain'];qc=prior['peptide_chain'];oldkey=prior['old_structure_pair_key'];provenance={'source':source,'source_record_id':rid};context='INHERITED_AUDITED_COMPLEX_FRAME'
        structures.append(dict(observation_id=oid,pair_id=pid,protein_id=p,peptide_id=q,source=source,source_record_id=rid,pdb_id=r['pdb_id'],protein_chains=pc,peptide_chains=qc,coordinate_assets=coords,inherited_labels=labs,inherited_maps=maps,original_split=r['original_split'] or None,original_structure_pair_key=oldkey,coordinate_context=context,label_status=r['label_status'],provenance=provenance,joint_link_ids=[x['link_id'] for x in joint]))
        if len(structures)%5000==0:print('catalog structures',len(structures),'assets',len(assets),flush=True)
    # Remove unresolved placeholder entities superseded by verified joint identity.
    usedq={r['peptide_id'] for r in quant+structures};chemicals={k:v for k,v in chemicals.items() if k in usedq}
    for name,rs in [('protein_sequences',prots.values()),('protein_components',complexes.values()),('peptide_sequences',peps.values()),('peptide_chemical_entities',chemicals.values()),('pairs',pairs.values())]:jsonl(out/'libraries'/f'{name}.jsonl',rs)
    jsonl(out/'supervision/quantitative.jsonl',quant);jsonl(out/'supervision/structural.jsonl',structures)
    # Public link table retains evidence, removes machine-specific paths/full-text copies.
    fields=['measurement_id','structure_id','link_id','current_evidence_tier','assayed_construct_status','assayed_construct_confirmed','biological_sample_link_confirmed','primary_pmid','source_url','primary_source_url','protein_mapping','peptide_mapping','pair_label_scope']
    jsonl(out/'supervision/verified_links.jsonl',({k:l.get(k) for k in fields} for l in links))
    jsonl(out/'staging/asset_transfer.jsonl',assets.values())
    # Deduplicated FASTA inputs for actual remote homology searches.
    for name,rr in [('protein',prots.values()),('peptide',peps.values())]:
        with (out/'splits'/f'{name}.fasta').open('w') as f:
            for r in sorted(rr,key=lambda x:x['id']):f.write('>'+r['id']+'\n'+r['sequence'].replace('J','X')+'\n')
    result=dict(quantitative_observations=len(quant),structure_observations=len(structures),unique_pairs=len(pairs),unique_protein_sequences=len(prots),protein_component_sets=len(complexes),unique_peptide_sequences=len(peps),peptide_chemical_entities=len(chemicals),verified_joint_links=len(links),source_structures=dict(collections.Counter(r['source'] for r in structures)),quantitative_endpoints=dict(collections.Counter(r['measurement']['endpoint'] for r in quant)),assets=len(assets),asset_bytes=sum(r['bytes'] for r in assets.values()),status='CATALOG_ONLY_LABEL_AND_SPLIT_GATES_PENDING')
    save(out/'audit/CATALOG_RECEIPT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
