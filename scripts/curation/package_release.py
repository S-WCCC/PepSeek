"""Seal and package a checked data release; metadata and structure libraries separate."""
import argparse,collections,concurrent.futures,hashlib,json,os,tarfile,time
from pathlib import Path
from build_catalog import sha,save,jsonl,loadjl,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--code-commit',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    acceptance=json.loads((root/'audit/ACCEPTANCE.json').read_text());assert acceptance['status']=='PASS'
    files=[p for p in root.rglob('*') if p.is_file() and p.parts[len(root.parts)] not in ['staging'] and p.name not in ['RELEASE_SEAL.json','RELEASE_MANIFEST.jsonl']]
    def record(p):return dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:manifest=list(pool.map(record,sorted(files)))
    jsonl(root/'RELEASE_MANIFEST.jsonl',manifest)
    seal=dict(version='curated_dual_v1_20261013',status='SEALED',code_commit=a.code_commit,model_dataset_sha256=acceptance['model_dataset_sha256'],manifest_sha256=sha(root/'RELEASE_MANIFEST.jsonl'),files=len(manifest),file_bytes=sum(x['bytes'] for x in manifest),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),historical_sources_immutable=True)
    if (root/'RELEASE_SEAL.json').exists():raise RuntimeError('Release already sealed')
    save(root/'RELEASE_SEAL.json',seal)
    # Public clean release contains every model label, map, split and provenance
    # record needed to train or inspect it. Raw ledgers stay on the development disk.
    sets={'clean_training':[],'structure_libraries':[]}
    retained_structures={r['path'] for side in ['protein','peptide'] for r in loadjl(root/f'libraries/{side}_structure_index.jsonl')}
    for r in manifest:
        p=r['path'];parts=Path(p).parts
        if p in retained_structures:sets['structure_libraries'].append(p)
        elif parts[0] in ['clean','supervision','maps'] or (parts[0]=='libraries' and len(parts)==2) or (parts[0]=='splits' and len(parts)==2) or (parts[0]=='audit' and len(parts)==2 and p.endswith(('.json','.jsonl'))) or p in ['README_ZH.md','index.html','DATA_CARD.json']:sets['clean_training'].append(p)
    packs=[]
    for name,paths in sets.items():
        target=out/(name+'.tar.gz')
        with tarfile.open(target,'w:gz',compresslevel=3) as tar:
            for p in paths:tar.add(root/p,arcname=p,recursive=False)
            for p in ['RELEASE_SEAL.json','RELEASE_MANIFEST.jsonl']:tar.add(root/p,arcname=p)
        size=target.stat().st_size;whole=sha(target);parts=[]
        with target.open('rb') as stream:
            for i,data in enumerate(iter(lambda:stream.read(8<<20),b'')):
                part=out/f'{name}.part{i:04d}';part.write_bytes(data);parts.append(dict(file=part.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        packs.append(dict(name=name,archive=target.name,bytes=size,sha256=whole,parts=parts));print(name,size,len(parts),flush=True)
    save(out/'DOWNLOAD_MANIFEST.json',dict(release=seal,archives=packs,source_coordinate_acquisition_manifest='audit/asset_manifest.jsonl',raw_source_snapshots_location='development1 data root / raw',raw_source_archives_not_required_for_sequence_model_forward=True))
if __name__=='__main__':main()
