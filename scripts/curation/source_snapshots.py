"""Preserve complete source ledgers separately from model-eligible observations."""
import argparse,collections,csv,gzip,json,shutil,tarfile
from pathlib import Path
from build_catalog import save,sha,jsonl,rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--prior',required=True);ap.add_argument('--curated',required=True);ap.add_argument('--integrated',required=True);a=ap.parse_args()
    root,prior,cur,base=map(Path,(a.root,a.prior,a.curated,a.integrated));files=[]
    selected=[(prior/'00_manifest/source_registry.json','raw/source_registry_historical.json'),
      (prior/'02_normalized/propedia26/propedia26_pairs_normalized.csv','raw/Propedia/source_rows.csv'),
      (prior/'02_normalized/pepbdb/pepbdb_pairs_normalized.csv','raw/PepBDB/source_rows.csv'),
      (base/'data/integrated_structure_source_registry.csv','raw/snapshots/integrated_structure_source_registry.csv'),
      (cur/'data/combined_structure_inventory.csv','raw/snapshots/eligible_structure_inventory.csv'),
      (cur/'data/confirmed_joint_links.csv','raw/snapshots/confirmed_joint_links.csv'),
      (prior/'08_benchmark_publication/splits/s4_publication_benchmark_manifest.csv','raw/snapshots/historical_s4_split.csv')]
    for src,rel in selected:
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);files.append(dict(path=rel,bytes=dest.stat().st_size,sha256=sha(dest)))
    for p in sorted((root/'raw').rglob('*')):
        if p.is_file() and p.relative_to(root).as_posix() not in {r['path'] for r in files}:files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    # Coverage counts are source rows, not unique interactions or independent labels.
    coverage=dict(ppikb_main_rows=21845,ppikb_branch_rows=3180,ppikb_structure_rows=18005,ppikb_unique_measurement_records=21936,ppikb_explicit_derived_structure_views=412,propedia_source_rows=sum(1 for _ in rows(root/'raw/Propedia/source_rows.csv')),pepbdb_source_rows=sum(1 for _ in rows(root/'raw/PepBDB/source_rows.csv')),prior_structure_candidate_rows=63320,prior_structure_label_eligible_rows=58715,prior_structure_excluded_rows=4605)
    assets=[]
    with (root/'staging/asset_transfer.jsonl').open() as f:
        for line in f:
            r=json.loads(line);assets.append({k:r[k] for k in ['path','sha256','bytes','urls']})
    jsonl(root/'audit/asset_manifest.jsonl',assets)
    save(root/'audit/SOURCE_COVERAGE.json',dict(counts=coverage,files=files,raw_retains_all_rows=True,local_source_paths_are_provenance_only=True))
    out=root/'staging/source_snapshots.tar.gz'
    with tarfile.open(out,'w:gz',compresslevel=3) as t:
        for r in files:t.add(root/r['path'],arcname=r['path'])
        for rel in ['audit/SOURCE_COVERAGE.json','audit/asset_manifest.jsonl']:t.add(root/rel,arcname=rel)
    save(root/'audit/SOURCE_PACKAGE_RECEIPT.json',dict(path=out.name,bytes=out.stat().st_size,sha256=sha(out),**coverage));print(out.stat().st_size,sha(out),flush=True)
if __name__=='__main__':main()
