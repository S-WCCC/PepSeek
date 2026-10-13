"""Bounded CPU label jobs start only after their exact source assets are present."""
import argparse,collections,concurrent.futures,json,time,os
from pathlib import Path
from build_catalog import loadjl,save
from structure_labels import init,process_group

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();root=Path(a.root)
    for x in ['labels','maps','audit/structure_rows','libraries/protein_structures','libraries/peptide_structures']:(root/x).mkdir(parents=True,exist_ok=True)
    rows=loadjl(root/'supervision/structural.jsonl');groups=collections.defaultdict(list)
    for r in rows:groups[tuple(x['sha256'] for x in r['coordinate_assets'])].append(r)
    pending=list(groups.values());active={};counts=collections.Counter();reasons=collections.Counter();began=time.time();last=-1;lastprint=0
    save(root/'audit/structure_launch_identity.json',dict(pid=os.getpid(),started=began,cmdline=Path('/proc/self/cmdline').read_bytes().decode().replace('\0',' '),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(root),)) as pool:
        while pending or active:
            remaining=[]
            for g in pending:
                if len(active)>=a.workers*2:remaining.append(g);continue
                required={x['path'] for r in g for field in ['coordinate_assets','inherited_maps'] for x in r[field]}
                if all((root/p).is_file() for p in required):active[pool.submit(process_group,g)]=len(g)
                else:remaining.append(g)
            pending=remaining
            done,_=concurrent.futures.wait(active,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED) if active else ([],[])
            for f in done:
                n=active.pop(f);result=f.result()
                assert len(result)==n
                counts.update(x['status'] for x in result);reasons.update(x['reason'] for x in result if x['reason'])
            completed=sum(counts.values())
            if completed!=last or time.time()-lastprint>60:
                save(root/'audit/structure_progress.json',dict(completed=completed,total=len(rows),status_counts=dict(counts),reasons=dict(reasons),active_groups=len(active),awaiting_asset_groups=len(pending),elapsed_seconds=time.time()-began))
                last=completed
            if time.time()-lastprint>60:print(completed,len(rows),dict(counts),'waiting groups',len(pending),flush=True);lastprint=time.time()
            if not active:time.sleep(10)
    save(root/'audit/STRUCTURE_LABEL_RECEIPT.json',dict(completed=sum(counts.values()),total=len(rows),status_counts=dict(counts),reasons=dict(reasons),elapsed_seconds=time.time()-began))
if __name__=='__main__':main()
