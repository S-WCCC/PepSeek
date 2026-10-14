"""Recover exact ATOM-order / unambiguous alignment positions after initial audit."""
import argparse,collections,concurrent.futures,json,shutil,time
from pathlib import Path
from build_catalog import loadjl,save,sha
from structure_labels import init,process_group

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=16);ap.add_argument('--history-tag',default='initial_mapping');a=ap.parse_args();root=Path(a.root)
    assert (root/'audit/STRUCTURE_LABEL_RECEIPT.json').exists(),'Initial audit must finish before recovery'
    if (root/'RELEASE_SEAL.json').exists():raise RuntimeError('Release is sealed')
    old=root/'audit/structure_history'/a.history_tag;old.mkdir(parents=True,exist_ok=True)
    if not (old/'RECEIPT.json').exists():shutil.copy2(root/'audit/STRUCTURE_LABEL_RECEIPT.json',old/'RECEIPT.json')
    selected=[];groups=collections.defaultdict(list)
    for row in loadjl(root/'supervision/structural.jsonl'):
        p=root/'audit/structure_rows'/(row['observation_id']+'.json');prior=old/p.name;r=json.loads((p if p.exists() else prior).read_text())
        if r['status']=='EXCLUDED' and (r['reason'] in ['No unique acceptable residue mapping','No mapped heavy atoms'] or r['reason'].startswith('Chain identity ambiguous:')):
            dest=old/p.name
            if not dest.exists():shutil.copy2(p,dest)
            if p.exists():assert sha(dest)==sha(p);p.unlink()
            selected.append(row);groups[tuple(x['sha256'] for x in row['coordinate_assets'])].append(row)
    counts=collections.Counter();started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(root),)) as pool:
        for result in pool.map(process_group,groups.values()):
            counts.update(x['status'] for x in result)
            save(root/'audit/mapping_recovery_progress.json',dict(total=len(selected),done=sum(counts.values()),counts=dict(counts)))
    allrows=[json.loads(p.read_text()) for p in (root/'audit/structure_rows').glob('*.json')]
    receipt=dict(completed=len(allrows),total=len(allrows),status_counts=dict(collections.Counter(x['status'] for x in allrows)),reasons=dict(collections.Counter(x['reason'] for x in allrows if x['reason'])),mapping_recovery=dict(attempted=len(selected),result=dict(counts)),elapsed_seconds=time.time()-started)
    save(root/'audit/STRUCTURE_LABEL_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
