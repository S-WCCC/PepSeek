"""Mapped heavy-atom geometry helpers, adapted from sealed PPIKB curation.
Source SHA recorded in data provenance. Functions do not mutate source assets.
"""
import collections,itertools,re,math,hashlib
import gemmi,numpy as np
from scipy.spatial import cKDTree
from Bio.Align import PairwiseAligner
AA=set("ACDEFGHIKLMNPQRSTVWY")
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def cat(b,n):
 c=b.get_mmcif_category(n);return [dict(zip(c,v)) for v in zip(*c.values())] if c else []
AA3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
SIDE={'A':'CB','R':'CB CG CD NE CZ NH1 NH2','N':'CB CG OD1 ND2','D':'CB CG OD1 OD2','C':'CB SG','Q':'CB CG CD OE1 NE2','E':'CB CG CD OE1 OE2','G':'','H':'CB CG ND1 CD2 CE1 NE2','I':'CB CG1 CG2 CD1','L':'CB CG1 CG2 CD1 CD2','K':'CB CG CD CE NZ','M':'CB CG SD CE','F':'CB CG CD1 CD2 CE1 CE2 CZ','P':'CB CG CD','S':'CB OG','T':'CB OG1 CG2','W':'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2','Y':'CB CG CD1 CD2 CE1 CE2 CZ OH','V':'CB CG1 CG2'}
SIDE['I']='CB CG1 CG2 CD1';SIDE['L']='CB CG CD1 CD2'

def mapping(source,entity):
 if not source or not entity:return None
 if source==entity:return [i if aa in AA else -1 for i,aa in enumerate(entity)],'EXACT'
 if entity in source:
  off=source.find(entity)
  if source.find(entity,off+1)>=0:return None
  return [off+i if aa in AA else -1 for i,aa in enumerate(entity)],'EXACT_FRAGMENT'
 if source in entity:
  off=entity.find(source)
  if entity.find(source,off+1)>=0:return None
  return [j-off if off<=j<off+len(source) and entity[j] in AA else -1 for j in range(len(entity))],'EXACT_SOURCE_WITH_COORDINATE_FLANKS_MASKED'
 if len(source)*len(entity)>20000000:return None
 al=PairwiseAligner(mode='global',match_score=2,mismatch_score=-3,open_gap_score=-8,extend_gap_score=-.5)
 aa=al.align(source,entity);a=aa[0]
 try:
  if not np.array_equal(aa[1].coordinates,a.coordinates):return None
 except IndexError:pass
 mp=[-1]*len(entity);aligned=0;matched=0
 for (s0,s1),(e0,e1) in zip(a.aligned[0],a.aligned[1]):
  for i,j in zip(range(s0,s1),range(e0,e1)):
   aligned+=1
   if source[i]==entity[j] and source[i] in AA:mp[j]=i;matched+=1
 if aligned<min(5,len(source),len(entity)) or matched/max(1,aligned)<.95 or matched/max(1,min(len(source),len(entity)))<.8:return None
 return mp,'UNIQUE_ALIGNMENT_EXACT_RESIDUES_ONLY'

def chain_data(b):
 ep={r['entity_id']:r for r in cat(b,'_entity_poly.')};mons=collections.defaultdict(dict)
 for r in cat(b,'_entity_poly_seq.'):
  try:mons[r['entity_id']][int(r['num'])-1]=r['mon_id']
  except ValueError:pass
 ch={}
 for r in cat(b,'_struct_asym.'):
  if r['entity_id'] not in ep:continue
  e=ep[r['entity_id']];s=re.sub(r'\s+','',e['pdbx_seq_one_letter_code_can']);lab=r['id']
  ch[lab]=dict(sequence=s,entity_id=r['entity_id'],type=e['type'],monomers=[mons[r['entity_id']].get(i,'UNK') for i in range(len(s))],auth=set(),atoms=[{} for _ in s],residue_ids={})
 for r in cat(b,'_pdbx_poly_seq_scheme.'):
  if r['asym_id'] in ch and r.get('pdb_strand_id'):ch[r['asym_id']]['auth'].add(r['pdb_strand_id'])
 tags=['label_asym_id','auth_asym_id','label_seq_id','auth_seq_id','label_comp_id','label_atom_id','label_alt_id','pdbx_PDB_ins_code','type_symbol','Cartn_x','Cartn_y','Cartn_z','occupancy','pdbx_PDB_model_num']
 first=None
 for rr in b.find('_atom_site.',tags):
  lab,auth,sid,asid,mon,name,alt,ins,el,x,y,z,occ,mo=list(rr)
  if first is None:first=mo
  if mo!=first or lab not in ch or el.upper() in {'H','D'}:continue
  c=ch[lab];c['auth'].add(auth)
  try:i=int(sid)-1;occup=float(occ);xyz=[float(x),float(y),float(z)]
  except ValueError:continue
  if not 0<=i<len(c['sequence']) or occup<=0 or not all(math.isfinite(v) for v in xyz):continue
  # Select one conformer per residue, rather than mixing alternate conformations.
  key=(name,alt);old=c['atoms'][i].get(key)
  if old is None or occup>old[0]:c['atoms'][i][key]=(occup,xyz,mon)
  c['residue_ids'][i]=dict(label_asym_id=lab,label_seq_id=i+1,auth_asym_id=auth,auth_seq_id=asid,insertion_code=ins)
 for c in ch.values():
  atoms=[];complete=[]
  for i,entries in enumerate(c['atoms']):
   altweights=collections.defaultdict(float)
   for (_,alt),(occ,_,_) in entries.items():
    if alt not in {'.','?',''}:altweights[alt]+=occ
   selected=sorted(altweights,key=lambda a:(-altweights[a],a!='A',a))[0] if altweights else ''
   selected_atoms={name:v for (name,alt),v in entries.items() if alt in {'.','?','',selected}}
   atoms.append(selected_atoms)
   aa=AA3.get(c['monomers'][i]);need=set(('N CA C O '+SIDE.get(aa,'')+(' OXT' if i==len(c['sequence'])-1 else '')).split())
   complete.append(aa is not None and aa==c['sequence'][i] and need<=set(selected_atoms))
  c['atoms']=atoms;c['complete']=complete
 return ch,first

def group_options(raw,ch,source):
 groups=[]
 for group in raw.split(';'):
  found=[]
  for token in group.split('/'):
   ma=re.fullmatch(r'\s*([^\s\[\]]+)\s*(?:\[\s*auth\s+([^\]]+)\])?\s*',token)
   if not ma:found=[];break
   lab,au=ma[1],ma[2]
   if lab in ch and (not au or au in ch[lab]['auth']):found.append(lab)
   else:
    matches=[k for k,c in ch.items() if (au or lab) in c['auth']]
    if len(matches)!=1:found=[];break
    found+=matches
  if found and len(set(found))==len(found):groups.append(tuple(found))
 if groups:return list(dict.fromkeys(groups)),'DECLARED_CHAIN'
 # A failed chain field may be repaired only by sequence identity, never by proximity alone.
 comps=source.split('/');matches=[]
 for s in comps:
  hits=[k for k,c in ch.items() if c['type'].startswith('polypeptide') and (s==c['sequence'] or (min(len(s),len(c['sequence']))>=10 and (s in c['sequence'] or c['sequence'] in s)))]
  if not hits:return [],'CHAIN_MAPPING_UNRESOLVED'
  matches.append(hits)
 for labs in itertools.product(*matches):
  if len(set(labs))==len(labs):groups.append(tuple(labs))
 if len(groups)>100:return [],'TOO_MANY_SEQUENCE_EQUIVALENT_CHAIN_OPTIONS'
 return groups,'SEQUENCE_MATCH_CHAIN_REPAIR'

def assembly_ops(b,ch):
 st=gemmi.make_structure_from_block(b);out={}
 for a in st.assemblies:
  amap=collections.defaultdict(dict)
  for g in a.generators:
   labs=list(g.subchains) or [k for k,c in ch.items() if c['auth']&set(g.chains)]
   for op in g.operators:
    mat=np.array(op.transform.mat.tolist());vec=np.array(op.transform.vec.tolist());key=tuple(np.round(np.r_[mat.ravel(),vec],8))
    for lab in labs:amap[lab][key]=(mat,vec,op.name)
  out[a.name]=amap
 return out

def group_maps(labs,ch,source):
 src=source.split('/')
 if len(src)!=len(labs):return None
 result=[]
 for lab,s in zip(labs,src):
  c=ch[lab]
  if not c['type'].startswith('polypeptide'):return None
  mp=mapping(s,c['sequence'])
  if mp is None:return None
  result.append((lab,s,mp[0],mp[1]))
 return result

def group_atoms(gm,ch,trans):
 mat,vec,_=trans;coords=[];resix=[];info=[];complete=[];seq=[];mapped=[]
 for lab,source,mp,kind in gm:
  c=ch[lab]
  for i,aa in enumerate(c['sequence']):
   j=len(info);a=c['atoms'][i];allowed=mp[i]>=0
   info.append(dict(chain=lab,entity_index_0based=i,source_component_index_0based=mp[i],source_component_sequence_sha256=digest(source),mapping=kind,monomer=c['monomers'][i],residue_id=c['residue_ids'].get(i,{})))
   seq.append(aa);mapped.append(allowed)
   complete.append(allowed and c['complete'][i])
   if allowed:
    for v in a.values():coords.append(np.array(v[1])@mat.T+vec);resix.append(j)
 return np.array(coords).reshape(-1,3),np.array(resix,dtype=np.int32),info,np.array(complete),''.join(seq),np.array(mapped)

def pair_metrics(gp,gq,ch,tp,tq):
 p=group_atoms(gp,ch,tp);q=group_atoms(gq,ch,tq)
 if not len(p[0]) or not len(q[0]):return None
 dp,_=cKDTree(q[0]).query(p[0],workers=1);dq,_=cKDTree(p[0]).query(q[0],workers=1)
 return dict(mind=float(dp.min()),pcontacts=len(set(p[1][dp<=5])),qcontacts=len(set(q[1][dq<=5])),p=p,q=q)

