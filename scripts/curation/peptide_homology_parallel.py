"""Query-sharded BLASTP-short search with streamed, retained qualifying alignments.

Protein fragments: 30% identity, 80% shorter-chain coverage.
Peptides: 50% identity, 80% coverage of BOTH sequences. A tiny motif shared with
a much longer peptide is not sufficient to define whole-peptide similarity.
"""
import argparse,concurrent.futures,csv,hashlib,json,subprocess,time,os
from pathlib import Path
from build_catalog import save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=20);a=ap.parse_args()
    root=Path(a.root);w=root/'splits/homology';parts=w/'peptide_parts';parts.mkdir(exist_ok=True)
    if (w/'RECEIPT.json').exists():raise RuntimeError('Homology sealed')
    records=[]
    for record in (root/'splits/peptide.fasta').read_text().split('>')[1:]:
        lines=record.splitlines();records.append((lines[0],''.join(lines[1:])))
    # Greedy balance total residues among independent query chunks.
    buckets=[[] for _ in range(a.workers)];loads=[0]*a.workers
    for r in sorted(records,key=lambda x:(-len(x[1]),x[0])):
        i=min(range(a.workers),key=lambda j:loads[j]);buckets[i].append(r);loads[i]+=len(r[1])
    for i,b in enumerate(buckets):(parts/f'q{i:02d}.fasta').write_text(''.join('>'+k+'\n'+s+'\n' for k,s in b))
    began=time.time();save(w/'parallel_identity.json',dict(pid=os.getpid(),cmdline=Path('/proc/self/cmdline').read_bytes().decode().replace('\0',' '),started=began,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def worker(i):
        receipt=parts/f'{i:02d}.receipt.json';query=parts/f'q{i:02d}.fasta';out=parts/f'{i:02d}.filtered.tsv'
        if receipt.exists():return json.loads(receipt.read_text())
        cmd=['blastp','-task','blastp-short','-query',str(query),'-db',str(w/'peptide_db'),'-outfmt','6 qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore','-evalue','1000000','-seg','no','-comp_based_stats','0','-num_threads','1','-max_target_seqs','1000000']
        with (parts/f'{i:02d}.stderr.log').open('w') as err:
            p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,text=True)
            save(parts/f'{i:02d}.identity.json',dict(pid=p.pid,command=cmd,start=time.time(),query_sha256=sha(query)))
            raw=kept=0;streamhash=hashlib.sha256()
            with out.open('w') as f:
                for line in p.stdout:
                    raw+=1;streamhash.update(line.encode());c=line.rstrip().split('\t')
                    if c[0]>=c[1] or float(c[2])<50:continue
                    qcov=(abs(int(c[7])-int(c[6]))+1)/int(c[4]);tcov=(abs(int(c[9])-int(c[8]))+1)/int(c[5])
                    if min(qcov,tcov)<.8:continue
                    f.write(line);kept+=1
            if p.wait()!=0:raise RuntimeError('BLAST shard failed '+str(i))
        r=dict(shard=i,raw_hsps=raw,retained_hsps=kept,raw_stream_sha256=streamhash.hexdigest(),filtered_sha256=sha(out));save(receipt,r);return r
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(worker,range(a.workers)):
            results.append(r);print('shard complete',r,flush=True)
    stats={}
    for kind,files,threshold in [('protein',[w/'protein_raw.tsv'],30),('peptide',sorted(parts.glob('*.filtered.tsv')),50)]:
        seen=set();raw=0
        with (w/(kind+'_edges.tsv')).open('w') as out:
            out.write('query\ttarget\tidentity_percent\tshorter_coverage\talignment_length\n')
            for file in files:
                with file.open() as f:
                    for line in f:
                        c=line.rstrip().split('\t');raw+=1;q,t=c[:2]
                        if q==t:continue
                        qcov=(abs(int(c[7])-int(c[6]))+1)/int(c[4]);tcov=(abs(int(c[9])-int(c[8]))+1)/int(c[5])
                        cov=(qcov if int(c[4])<=int(c[5]) else tcov) if kind=='protein' else min(qcov,tcov)
                        key=tuple(sorted((q,t)))
                        if float(c[2])>=threshold and cov>=.8 and key not in seen:
                            seen.add(key);out.write('\t'.join([*key,c[2],str(cov),c[3]])+'\n')
        stats[kind]=dict(edges=len(seen),filtered_input_rows=raw,minimum_identity_percent=threshold,minimum_coverage=.8,coverage_policy='shorter' if kind=='protein' else 'both',edge_sha256=sha(w/(kind+'_edges.tsv')))
    save(w/'RECEIPT.json',dict(status='COMPLETED',searches=stats,peptide_shards=results,elapsed_seconds=time.time()-began,inputs={x:sha(root/'splits'/(x+'.fasta')) for x in ['protein','peptide']},parameters_locked_before_partition_assignment=True))
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
