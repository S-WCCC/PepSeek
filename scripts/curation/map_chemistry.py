import argparse,json,collections
from pathlib import Path
from build_catalog import loadjl,jsonl,save
from chemical_mapping import mapped_identity

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    entities=loadjl(root/'libraries/peptide_chemical_entities.jsonl');result=[];counts=collections.Counter()
    for i,e in enumerate(entities):
        r={'id':e['id'],'sequence':e['sequence'],'peptide_sequence_id':e['peptide_sequence_id']}
        if e['chemistry_mode']=='unknown':r.update(chemistry_mode='unknown',chemical_identity_id=e['id'],mapping_status='SEQUENCE_ONLY_CHEMISTRY_MASKED')
        else:
            try:r.update(mapped_identity(e));r['mapping_status']='EXPLICIT_CHEMISTRY_MAPPED'
            except Exception as err:r.update(mapping_status='HOLD_CHEMICAL_RESIDUE_MAP',reason=str(err))
        result.append(r);counts[r['mapping_status']]+=1
        if i%5000==0:print(i,dict(counts),flush=True)
    jsonl(root/'libraries/model_peptide_entities.jsonl',result);save(root/'audit/CHEMISTRY_MAPPING_RECEIPT.json',dict(counts=counts,hold_reasons=dict(collections.Counter(r.get('reason') for r in result if r.get('reason')))))
    print(dict(counts),flush=True)
if __name__=='__main__':main()
