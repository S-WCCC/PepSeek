"""Prepare relocatable metadata and source-verified coordinate acquisition lists."""
import argparse,json,os,shutil,tarfile,collections
from pathlib import Path
from build_catalog import loadjl,jsonl,save,rows,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--integrated',required=True);a=ap.parse_args()
    root=Path(a.root);base=Path(a.integrated);assets=loadjl(root/'staging/asset_transfer.jsonl')
    urlbysha={r['sha256']:r['source_url'] for r in rows(base/'audit/coordinate_manifest.csv') if r['sha256'] and r['source_url']}
    roles={}
    for r in assets:
        src=Path(r['source_path']);dest=root/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            try:os.link(src,dest)
            except OSError:shutil.copy2(src,dest)
        urls=[]
        if r['path'].endswith('.cif.gz'):
            pdb=src.name.split('.')[0].lower()
            if r['sha256'] in urlbysha:urls.append(urlbysha[r['sha256']])
            if len(pdb)==4:urls.extend([f'https://files.rcsb.org/download/{pdb}.cif.gz',f'https://files.wwpdb.org/pub/pdb/data/structures/obsolete/mmCIF/{pdb[1:3]}/{pdb}.cif.gz'])
        r['urls']=list(dict.fromkeys(urls));roles[r['path']]='receptor' if src.name=='receptor.pdb' else 'peptide' if src.name=='peptide.pdb' else 'complex'
    jsonl(root/'staging/asset_transfer.jsonl',assets)
    rs=loadjl(root/'supervision/structural.jsonl')
    for r in rs:
        for x in r['coordinate_assets']:x['role']=roles[x['path']]
    jsonl(root/'supervision/structural.jsonl',rs)
    # Source workbook snapshots kept on the data disk, separate from clean tables.
    raw=root/'raw/PPIKB';raw.mkdir(exist_ok=True)
    for name in ['main.xlsx','branch.xlsx','structure.xlsx','sources.json']:shutil.copy2(base/'sources'/name,raw/name)
    # Small source-derived assets have no archive URL and are transferred directly.
    local=[r for r in assets if not r['urls']]
    out=root/'staging/local_assets.tar.gz'
    with tarfile.open(out,'w:gz',compresslevel=3) as tar:
        for r in local:tar.add(root/r['path'],arcname=r['path'])
    save(root/'audit/STAGING_RECEIPT.json',dict(assets=len(assets),coordinate_bytes=sum(r['bytes'] for r in assets),remote_downloadable=sum(bool(r['urls']) for r in assets),locally_packaged=len(local),local_archive_bytes=out.stat().st_size,local_archive_sha256=sha(out)))
    print((root/'audit/STAGING_RECEIPT.json').read_text(),flush=True)
if __name__=='__main__':main()
