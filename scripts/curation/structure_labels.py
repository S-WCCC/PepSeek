"""Rebuild interface/contact supervision and per-chain libraries from coordinates.

Experimental coordinates stay in labels/libraries, never sequence model inputs.
Positive contacts require observed mapped atoms; negatives require complete
standard heavy atoms on both residues. Incomplete/unmapped labels are masked.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,collections,concurrent.futures,gzip,json,time,traceback
from pathlib import Path
import numpy as np,gemmi
from scipy.spatial.distance import cdist
from geometry_helpers import chain_data,group_maps,group_atoms,assembly_ops,AA3,SIDE,AA,mapping
from build_catalog import loadjl,jsonl,save,sha,digest
ROOT=None;PROTS=None;PEPS=None

def init(root):
    global ROOT,PROTS,PEPS
    ROOT=Path(root);PROTS={r['id']:r for r in loadjl(ROOT/'libraries/protein_components.jsonl')};PEPS={r['id']:r for r in loadjl(ROOT/'libraries/peptide_chemical_entities.jsonl')}

def pdb_chain(path):
    st=gemmi.read_structure(str(path));result={}
    for chain in st[0]:
        res=[r for r in chain if r.name in AA3 or gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        if not res:continue
        seq=''.join(AA3.get(r.name,gemmi.find_tabulated_residue(r.name).one_letter_code.upper() or 'X') for r in res)
        atoms=[];complete=[];rid={}
        for i,r in enumerate(res):
            weights=collections.defaultdict(float)
            for a in r:
                if a.altloc not in ('\x00',' ',''):weights[a.altloc]+=a.occ
            alt=sorted(weights,key=lambda x:(-weights[x],x))[0] if weights else ''
            aa={a.name:(a.occ,[a.pos.x,a.pos.y,a.pos.z],r.name) for a in r if a.element.name not in ('H','D') and a.occ>0 and a.altloc in ('\x00',' ','',alt)}
            need=set(('N CA C O '+SIDE.get(AA3.get(r.name),'')+(' OXT' if i==len(res)-1 else '')).split())
            atoms.append(aa);complete.append(r.name in AA3 and need<=set(aa));rid[i]={'auth_asym_id':chain.name,'auth_seq_id':str(r.seqid.num),'insertion_code':r.seqid.icode}
        result[chain.name]=dict(sequence=seq,entity_id=chain.name,type='polypeptide(L)',monomers=[r.name for r in res],auth={chain.name},atoms=atoms,complete=complete,residue_ids=rid)
    return result,'1'

def resolve(declared,comps,ch,namespace):
    tokens=declared.split('/')
    candidates=[]
    for i,s in enumerate(comps):
        token=tokens[i] if i<len(tokens) else ''
        if namespace=='label' and token in ch:candidates.append([token]);continue
        hits=[k for k,c in ch.items() if token in c['auth']]
        if len(hits)==1:candidates.append(hits);continue
        hits=[k for k,c in ch.items() if s==c['sequence']]
        if len(hits)!=1:raise ValueError('Chain identity ambiguous: '+declared)
        candidates.append(hits)
    labs=tuple(x[0] for x in candidates)
    if len(set(labs))!=len(labs):raise ValueError('Repeated component chain')
    gm=group_maps(labs,ch,'/'.join(comps))
    if gm is None:raise ValueError('No unique acceptable residue mapping')
    return gm

def project(gm,ch,trans):
    mat,vec,_=trans;xyz=[];indices=[];n=sum(len(x[1]) for x in gm);complete=np.zeros(n,bool);observed=np.zeros(n,bool);resmaps=[];offset=0;libraries=[]
    for lab,seq,mp,kind in gm:
        c=ch[lab];cx=[];ci=[];names=[];mons=[]
        for i,j in enumerate(mp):
            if j<0:continue
            if c['atoms'][i]:observed[offset+j]=True
            complete[offset+j]=c['complete'][i]
            resmaps.append(dict(component_label_asym_id=lab,coordinate_entity_index=i,input_residue_index=offset+j,component_residue_index=j,mapping=kind,monomer=c['monomers'][i],coordinate_residue_id=c['residue_ids'].get(i,{})))
            for name,v in sorted(c['atoms'][i].items()):
                coord=np.array(v[1])@mat.T+vec
                xyz.append(coord);indices.append(offset+j);cx.append(coord);ci.append(j);names.append(name);mons.append(c['monomers'][i])
        libraries.append(dict(chain=lab,sequence=seq,coordinates=np.asarray(cx,np.float32).reshape(-1,3),atom_residue=np.array(ci,np.int32),atom_names=np.asarray(names),monomers=np.asarray(mons),observed_mask=observed[offset:offset+len(seq)].copy(),complete_mask=complete[offset:offset+len(seq)].copy()))
        offset+=len(seq)
    return np.asarray(xyz).reshape(-1,3),np.asarray(indices,np.int32),complete,observed,resmaps,libraries

def process_group(rows):
    # Same source file parsed once for every chain pair that references it.
    r0=rows[0];paths=[ROOT/x['path'] for x in r0['coordinate_assets']]
    parsed=[]
    for path in paths:
        if path.name.endswith(('.cif','.cif.gz')):
            block=gemmi.cif.read(str(path)).sole_block();ch,mo=chain_data(block);parsed.append((ch,mo,block))
        else:
            ch,mo=pdb_chain(path);parsed.append((ch,mo,None))
    result=[]
    for r in rows:
        dest=ROOT/'labels'/(r['observation_id']+'.npz');receipt=ROOT/'audit/structure_rows'/(r['observation_id']+'.json')
        if receipt.exists():result.append(json.loads(receipt.read_text()));continue
        status=dict(observation_id=r['observation_id'],source_record_id=r['source_record_id'],status='EXCLUDED',reason=None)
        try:
            citations=[]
            for _,_,block in parsed:
                if block is not None:
                    cat=block.get_mmcif_category('_citation.')
                    if cat:
                        for c in (dict(zip(cat,v)) for v in zip(*cat.values())):
                            citations.append({k:c.get(k) for k in ['id','pdbx_database_id_PubMed','pdbx_database_id_DOI','year']})
            status['citations']=citations
            ps=PROTS[r['protein_id']]['components'];qs=[PEPS[r['peptide_id']]['sequence']]
            identity=(np.eye(3),np.zeros(3),'identity');tp=tq=identity
            if len(parsed)==1:
                ch,mo,b=parsed[0];ns='label' if r['source']=='PPIKB' else 'auth';gp=resolve(r['protein_chains'],ps,ch,ns);gq=resolve(r['peptide_chains'],qs,ch,ns);cp=cq=ch
                # Recovered records explicitly identify biological-assembly operators.
                chosen=None
                for item in r['inherited_maps']:
                    with gzip.open(ROOT/item['path'],'rt',encoding='utf-8') as f:mpdata=json.load(f)
                    if mpdata.get('assembly_id') and mpdata.get('protein_operator') is not None:chosen=mpdata;break
                if chosen:
                    ops=assembly_ops(b,ch).get(str(chosen['assembly_id']),{})
                    def op(g,key):
                        found=[v for v in ops.get(g[0][0],{}).values() if v[2]==key]
                        if len(found)!=1:raise ValueError('Assembly operator mapping unresolved')
                        return found[0]
                    tp=op(gp,chosen['protein_operator']);tq=op(gq,chosen['peptide_operator'])
                elif set(x[0] for x in gp)&set(x[0] for x in gq):raise ValueError('Same physical chain on both sides without distinct assembly operators')
            elif len(parsed)==2:
                # PepBDB separate receptor/peptide files retain the same complex frame.
                ri=next((i for i,p in enumerate(paths) if 'receptor' in str(p).lower()),None)
                # Content-addressed names lose original basename; asset role is explicit in catalog.
                ri=next((i for i,x in enumerate(r['coordinate_assets']) if x.get('role')=='receptor'),ri)
                if ri is None:ri=0
                cp,mo,_=parsed[ri];cq,_,_=parsed[1-ri]
                gp=resolve(r['protein_chains'],ps,cp,'auth');gq=resolve(r['peptide_chains'],qs,cq,'auth')
            else:raise ValueError('Unexpected coordinate file count')
            p=project(gp,cp,tp);q=project(gq,cq,tq)
            if not len(p[0]) or not len(q[0]):raise ValueError('No mapped heavy atoms')
            lp,lq=sum(map(len,ps)),sum(map(len,qs));dist=np.full((lp,lq),np.inf,np.float32)
            for k in range(0,len(p[0]),64):
                d=cdist(p[0][k:k+64],q[0]);np.minimum.at(dist,(np.repeat(p[1][k:k+64],len(q[0])),np.tile(q[1],len(p[0][k:k+64]))),d.ravel())
            minimum=float(dist.min())
            if minimum<.8:raise ValueError('Severe interchain atom overlap')
            if minimum>5:raise ValueError('No mapped observed interface within 5 A')
            neg=p[2][:,None]&q[2][None,:];contact=np.full((lp,lq),np.nan,np.float32)
            contact[dist<=8]=1;contact[neg&(dist>8)]=0
            pi=np.full(lp,np.nan,np.float32);qi=np.full(lq,np.nan,np.float32)
            pi[(contact==1).any(1)]=1;qi[(contact==1).any(0)]=1;pi[(contact==0).all(1)]=0;qi[(contact==0).all(0)]=0
            bins=np.full((lp,lq),-1,np.int8);valid=neg&np.isfinite(dist);bins[valid]=np.searchsorted([4.5,8,12,16],dist[valid],side='left')
            np.savez_compressed(dest,contact=contact,protein_interface=pi,peptide_interface=qi,distance_bins=bins)
            mapobj=dict(observation_id=r['observation_id'],protein_components=ps,peptide_components=qs,protein_map=p[4],peptide_map=q[4],model=mo,protein_operator={'name':tp[2],'matrix':tp[0].tolist(),'vector':tp[1].tolist()},peptide_operator={'name':tq[2],'matrix':tq[0].tolist(),'vector':tq[1].tolist()},coordinate_assets=r['coordinate_assets'],native_coordinates_are_supervision_only=True)
            mapfile=ROOT/'maps'/(r['observation_id']+'.json.gz')
            with gzip.open(mapfile,'wt',encoding='utf-8') as f:json.dump(mapobj,f,separators=(',',':'))
            librows=[]
            for side,data in [('protein',p),('peptide',q)]:
                for comp in data[5]:
                    h=digest([r['coordinate_assets'],comp['chain'],comp['sequence'],tp[2] if side=='protein' else tq[2],sha(mapfile)])
                    # Content key uses exact transformed atoms and mapping, permitting source aliases.
                    import hashlib
                    hh=hashlib.sha256(comp['coordinates'].tobytes()+comp['atom_residue'].tobytes()+comp['sequence'].encode()+comp['atom_names'].tobytes()).hexdigest()
                    rel=f'libraries/{side}_structures/{hh}.npz';path=ROOT/rel
                    if not path.exists():np.savez_compressed(path,**{k:v for k,v in comp.items() if isinstance(v,np.ndarray)})
                    librows.append(dict(id=side[0].upper()+'S_'+hh,side=side,path=rel,sequence=comp['sequence'],chain=comp['chain'],coordinate_sha256s=[x['sha256'] for x in r['coordinate_assets']],observation_id=r['observation_id'],map_path=mapfile.relative_to(ROOT).as_posix()))
            status.update(status='PASS',reason=None,label_path=dest.relative_to(ROOT).as_posix(),label_sha256=sha(dest),map_path=mapfile.relative_to(ROOT).as_posix(),map_sha256=sha(mapfile),protein_length=lp,peptide_length=lq,positive_contacts=int(np.sum(contact==1)),negative_contacts=int(np.sum(contact==0)),masked_contacts=int(np.isnan(contact).sum()),distance_labels=int(valid.sum()),minimum_distance_A=minimum,structure_libraries=librows,contact_fingerprint=digest([ps,qs,np.argwhere(contact==1).tolist()]))
        except Exception as e:status.update(reason=str(e),error_type=type(e).__name__)
        save(receipt,status);result.append(status)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=20);ap.add_argument('--limit',type=int);a=ap.parse_args()
    root=Path(a.root)
    for x in ['labels','maps','audit/structure_rows','libraries/protein_structures','libraries/peptide_structures']:(root/x).mkdir(parents=True,exist_ok=True)
    rs=loadjl(root/'supervision/structural.jsonl')
    if a.limit:rs=rs[:a.limit]
    groups=collections.defaultdict(list)
    for r in rs:groups[tuple(x['sha256'] for x in r['coordinate_assets'])].append(r)
    began=time.time();done=0;counts=collections.Counter();errors=collections.Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(root),)) as pool:
        futures={pool.submit(process_group,v):len(v) for v in groups.values()}
        for fut in concurrent.futures.as_completed(futures):
            try:result=fut.result()
            except Exception:
                traceback.print_exc();raise
            done+=len(result);counts.update(x['status'] for x in result);errors.update(x['reason'] for x in result if x['reason'])
            save(root/'audit/structure_progress.json',dict(completed=done,total=len(rs),status_counts=dict(counts),reasons=dict(errors),elapsed_seconds=time.time()-began))
            if done%100< len(result):print(done,len(rs),dict(counts),flush=True)
    save(root/'audit/STRUCTURE_LABEL_RECEIPT.json',dict(completed=done,total=len(rs),status_counts=dict(counts),reasons=dict(errors),elapsed_seconds=time.time()-began,software_acceptance_only=bool(a.limit)))
if __name__=='__main__':main()
