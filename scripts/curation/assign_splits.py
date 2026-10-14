"""Deterministic connected-component splits; neither labels nor model scores used.

Unions: protein homology, peptide similarity, chemical identity, exact paired
identity, verified measurement/structure links, shared PDB and source publication.
Historical split files are immutable provenance, not overwritten by this new version.
"""
import argparse,collections,csv,json,math,random
from pathlib import Path
from build_catalog import loadjl,jsonl,save,digest,sha

class Union:
    def __init__(self,keys):
        keys=list(keys);self.p={k:k for k in keys};self.n={k:1 for k in keys}
    def find(self,x):
        while x!=self.p[x]:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.n[a]<self.n[b]:a,b=b,a
        self.p[b]=a;self.n[a]+=self.n[b]

def make(root):
    root=Path(root);q=loadjl(root/'supervision/quantitative.jsonl');s=loadjl(root/'supervision/structural.jsonl');allrows=q+s
    proteins={r['id']:r for r in loadjl(root/'libraries/protein_components.jsonl')};peptides={r['id']:r for r in loadjl(root/'libraries/peptide_chemical_entities.jsonl')}
    seqs={kind:{r['id']:r for r in loadjl(root/f'libraries/{kind}_sequences.jsonl')} for kind in ['protein','peptide']}
    clusters={};audit={}
    for kind in seqs:
        uf=Union(seqs[kind]);file=root/f'splits/homology/{kind}_edges.tsv';count=0
        with file.open() as f:
            for edge in csv.DictReader(f,delimiter='\t'):uf.join(edge['query'],edge['target']);count+=1
        members=collections.defaultdict(list)
        for k in uf.p:members[uf.find(k)].append(k)
        clusters[kind]={i:kind[0].upper()+'H_'+digest(sorted(m))[:24] for m in members.values() for i in m}
        jsonl(root/f'splits/{kind}_clusters.jsonl',({'sequence_id':i,'cluster_id':c} for i,c in sorted(clusters[kind].items())))
        audit[kind]={'sequences':len(uf.p),'clusters':len(members),'alignment_edges':count,'largest_cluster':max(map(len,members.values())),'edges_sha256':sha(file)}
    uf=Union(r['observation_id'] for r in allrows);index={};reasons=collections.Counter()
    def attach(row,key):
        oid=row['observation_id']
        if key in index:uf.join(oid,index[key]);reasons[key[0]]+=1
        else:index[key]=oid
    for r in allrows:
        attach(r,('pair',r['pair_id']))
        for seqid in proteins[r['protein_id']]['component_ids']:attach(r,('protein_homology',clusters['protein'][seqid]))
        qid=peptides[r['peptide_id']]['peptide_sequence_id'];attach(r,('peptide_similarity',clusters['peptide'][qid]))
        chemical=peptides[r['peptide_id']]['chemical_identity']
        if chemical:attach(r,('chemical_graph',chemical['smiles']))
        if r.get('pdb_id'):attach(r,('pdb',r['pdb_id'].upper()))
        prov=r.get('provenance',{});pmid=str(prov.get('pmid') or '').strip();doi=str(prov.get('doi') or '').strip().lower()
        if pmid not in ('','None','null','nan'):attach(r,('publication_pmid',pmid))
        if doi not in ('','none','null','nan'):attach(r,('publication_doi',doi))
        receipt=root/'audit/structure_rows'/(r['observation_id']+'.json')
        if receipt.exists():
            rr=json.loads(receipt.read_text())
            for c in rr.get('citations',[]):
                pm=str(c.get('pdbx_database_id_PubMed') or '').strip();do=str(c.get('pdbx_database_id_DOI') or '').strip().lower()
                if pm not in ('','?','.','None'):attach(r,('publication_pmid',pm))
                if do not in ('','?','.','none'):attach(r,('publication_doi',do))
            if rr.get('contact_fingerprint'):attach(r,('exact_mapped_contact_fingerprint',rr['contact_fingerprint']))
    bysource={r['source_record_id']:r['observation_id'] for r in allrows}
    for l in loadjl(root/'supervision/verified_links.jsonl'):
        a,b=bysource.get(l['measurement_id']),bysource.get(l['structure_id'])
        if a and b:uf.join(a,b);reasons['verified_joint_link']+=1
    groups=collections.defaultdict(list)
    for r in allrows:groups[uf.find(r['observation_id'])].append(r)
    # Atomic components, balanced by observation counts; no label-value balancing.
    components=[]
    for items in groups.values():
        cid='C_'+digest(sorted(x['observation_id'] for x in items))[:24]
        components.append({'id':cid,'items':items,'n':len(items),'quantitative':sum(x['observation_id'].startswith('M_') for x in items)})
    components.sort(key=lambda x:(-x['n'],digest([17,x['id']])))
    targets={'train':.8*len(allrows),'valid':.1*len(allrows),'test':.1*len(allrows)};counts=collections.Counter();assignment={};comprows=[]
    for c in components:
        # Largest component goes to train even when it exceeds its target.
        choice='train' if not assignment else max(targets,key=lambda k:(targets[k]-counts[k],k=='train',k=='valid'))
        counts[choice]+=c['n']
        for r in c['items']:assignment[r['observation_id']]=(choice,c['id'])
        comprows.append({'component_id':c['id'],'split':choice,'observations':c['n'],'quantitative':c['quantitative']})
    out=[]
    for r in allrows:
        split,cid=assignment[r['observation_id']]
        out.append(dict(observation_id=r['observation_id'],pair_id=r['pair_id'],split=split,component_id=cid,supervision='quantitative' if r['observation_id'].startswith('M_') else 'structural',source=r['source'],original_split=r.get('original_split'),historical_split_reassigned_in_new_version=bool(r.get('original_split') and r.get('original_split')!=split)))
    jsonl(root/'splits/observations.jsonl',out);jsonl(root/'splits/components.jsonl',comprows)
    # Recheck every actual grouping key, including links, rather than counts only.
    percluster=collections.defaultdict(set)
    for r in allrows:
        split=assignment[r['observation_id']][0]
        for seqid in proteins[r['protein_id']]['component_ids']:percluster[('p',clusters['protein'][seqid])].add(split)
        percluster[('q',clusters['peptide'][peptides[r['peptide_id']]['peptide_sequence_id']])].add(split)
        percluster[('pair',r['pair_id'])].add(split)
        if r.get('pdb_id'):percluster[('pdb',r['pdb_id'])].add(split)
    crossing={str(k):list(v) for k,v in percluster.items() if len(v)>1}
    if crossing:raise AssertionError('Split leakage '+str(list(crossing.items())[:4]))
    report=dict(seed=17,nominal_ratios=[.8,.1,.1],actual_counts=dict(counts),components=len(components),largest_components=comprows[:20],sequence_clustering=audit,grouping_operations=dict(reasons),cross_partition_group_count=0,policy='dual_cold_homology_and_PDB_publication_components',historical_artifacts_modified=False,benchmark_novelty='New version for retraining; project has prior exposure to parts of this corpus. Not a new external independent confirmation set.',label_values_used_for_split=False)
    save(root/'audit/SPLIT_RECEIPT.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();make(a.root)
