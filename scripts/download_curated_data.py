"""Download a sealed PepSeek data release and verify every archive part.

Default: all model-ready tables, supervision labels, maps and split assignments.
Use --structures to also obtain the separate protein/peptide structure libraries.
"""
import argparse,concurrent.futures,hashlib,json,os,tarfile,time,urllib.request
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--tag',default='curated-dual-v1-20261013');ap.add_argument('--structures',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True);cache=root/'.downloads';cache.mkdir(exist_ok=True)
    base=f'https://github.com/S-WCCC/PepSeek/releases/download/{a.tag}/'
    req=urllib.request.Request(base+'DOWNLOAD_MANIFEST.json',headers={'User-Agent':'PepSeek-data/1'})
    with urllib.request.urlopen(req,timeout=90) as r:manifest=json.load(r)
    (cache/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    archives=[x for x in manifest['archives'] if x['name']=='clean_training' or a.structures]
    def get(part):
        p=cache/part['file']
        if p.exists() and p.stat().st_size==part['bytes'] and digest(p)==part['sha256']:return
        for attempt in range(4):
            try:
                tmp=p.with_name(p.name+'.download')
                with urllib.request.urlopen(urllib.request.Request(base+part['file'],headers={'User-Agent':'PepSeek-data/1'}),timeout=120) as f,tmp.open('wb') as o:
                    while True:
                        b=f.read(1<<20)
                        if not b:break
                        o.write(b)
                if tmp.stat().st_size!=part['bytes'] or digest(tmp)!=part['sha256']:raise ValueError('Downloaded part SHA mismatch')
                os.replace(tmp,p);return
            except Exception:
                if attempt==3:raise
                time.sleep(2+attempt)
    for ar in archives:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(get,ar['parts']))
        archive=cache/ar['archive']
        with archive.open('wb') as out:
            for part in ar['parts']:
                with (cache/part['file']).open('rb') as f:
                    for b in iter(lambda:f.read(1<<20),b''):out.write(b)
        assert archive.stat().st_size==ar['bytes'] and digest(archive)==ar['sha256']
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                if m.name.startswith('/') or '..' in Path(m.name).parts or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
                if m.isfile():
                    target=root/m.name;target.parent.mkdir(parents=True,exist_ok=True);b=tar.extractfile(m).read()
                    if target.exists():assert digest(target)==hashlib.sha256(b).hexdigest(),'Existing file belongs to a different release'
                    else:target.write_bytes(b)
        print('Verified and extracted',ar['name'],flush=True)
    seal=json.loads((root/'RELEASE_SEAL.json').read_text());assert digest(root/'RELEASE_MANIFEST.jsonl')==seal['manifest_sha256']
    for line in (root/'RELEASE_MANIFEST.jsonl').open():
        r=json.loads(line);p=root/r['path']
        if p.exists():assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['path']
    print('Ready:',root/'clean/dataset.json')
if __name__=='__main__':main()
