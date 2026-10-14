"""Actual all-versus-all homology searches with preserved commands and receipts.

MMseqs2 searches protein components. BLASTP-short searches peptide sequences,
including very short peptides that k-mer prefilters can miss. Protein edges use
>=80% shorter-component coverage; peptide edges use >=80% coverage of both.
No labels or measured values are read by this program.
"""
import argparse,csv,json,subprocess,time,hashlib,os
from pathlib import Path
from build_catalog import save,sha

def run(command,root):
    with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'command':command,'time':time.time()})+'\n')
    with (root/'search.stdout.log').open('a') as out,(root/'search.stderr.log').open('a') as err:
        subprocess.run(command,stdout=out,stderr=err,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--threads',type=int,default=20);ap.add_argument('--stage',choices=['all','protein'],default='all');a=ap.parse_args()
    root=Path(a.root);s=root/'splits';work=s/'homology';work.mkdir(exist_ok=True)
    if (work/'RECEIPT.json').exists():raise RuntimeError('Search already sealed')
    started=time.time();pid={'pid':os.getpid(),'started':started,'cmdline':Path('/proc/self/cmdline').read_bytes().decode().replace('\0',' '),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()};save(work/'identity.json',pid)
    run(['mmseqs','easy-search',str(s/'protein.fasta'),str(s/'protein.fasta'),str(work/'protein_raw.tsv'),str(work/'mmseqs_tmp'),'--threads',str(a.threads),'-s','7.5','--min-seq-id','0.30','--cov-mode','0','-c','0','--alignment-mode','3','--max-seqs','1000000','-e','1000','--format-output','query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits'],work)
    run(['makeblastdb','-in',str(s/'peptide.fasta'),'-dbtype','prot','-out',str(work/'peptide_db')],work)
    if a.stage=='protein':
        save(work/'PROTEIN_SEARCH_RECEIPT.json',dict(status='PROTEIN_SEARCH_COMPLETED',protein_raw_sha256=sha(work/'protein_raw.tsv'),next_step='peptide_homology_parallel.py'))
        return
    run(['blastp','-task','blastp-short','-query',str(s/'peptide.fasta'),'-db',str(work/'peptide_db'),'-out',str(work/'peptide_raw.tsv'),'-outfmt','6 qseqid sseqid pident length qlen slen qstart qend sstart send evalue bitscore','-evalue','1000000','-seg','no','-comp_based_stats','0','-num_threads',str(a.threads),'-max_target_seqs','1000000'],work)
    stats={}
    for kind,threshold in [('protein',30),('peptide',50)]:
        seen=set();raw=0
        with (work/(kind+'_raw.tsv')).open() as f,(work/(kind+'_edges.tsv')).open('w') as out:
            out.write('query\ttarget\tidentity_percent\tshorter_coverage\talignment_length\n')
            for line in f:
                c=line.rstrip().split('\t');raw+=1
                q,t=c[:2]
                if q==t:continue
                pid=float(c[2]);aln=int(c[3]);ql,tl=map(int,c[4:6]);qspan=abs(int(c[7])-int(c[6]))+1;tspan=abs(int(c[9])-int(c[8]))+1
                cov=(qspan/ql if ql<=tl else tspan/tl) if kind=='protein' else min(qspan/ql,tspan/tl)
                key=tuple(sorted([q,t]))
                if pid>=threshold and cov>=.8 and key not in seen:
                    seen.add(key);out.write('\t'.join([*key,str(pid),str(cov),str(aln)])+'\n')
        stats[kind]={'raw_hsps':raw,'edges':len(seen),'minimum_identity_percent':threshold,'minimum_coverage':.8,'coverage_policy':'shorter' if kind=='protein' else 'both','edge_sha256':sha(work/(kind+'_edges.tsv'))}
    save(work/'RECEIPT.json',{'elapsed_seconds':time.time()-started,'inputs':{k:sha(s/(k+'.fasta')) for k in ['protein','peptide']},'searches':stats,'status':'COMPLETED','scope':'Reported alignment edges; heuristic search recall is not a mathematical proof of absence of all evolutionary relatedness.'})
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
