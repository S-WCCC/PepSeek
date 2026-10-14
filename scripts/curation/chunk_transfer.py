"""Transfer an exact archive to the pinned development node using small chunks."""
import argparse,concurrent.futures,hashlib,json,subprocess,time,shlex,base64
from pathlib import Path

REMOTE_BASE='/data/PepSeek/runtime/curation/curated_dual_v1_20261013/transfers'

def remote(code):
    encoded=base64.b64encode(code.encode()).decode();arg="import base64;exec(base64.b64decode('"+encoded+"'))"
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','easydesign-cluster-two','python -c '+shlex.quote(arg)],capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    return p.stdout

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--file',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--extract-to');a=ap.parse_args()
    f=Path(a.file);cache=Path(a.cache)/f.name;cache.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();manifest=[]
    with f.open('rb') as stream:
        for i,data in enumerate(iter(lambda:stream.read(2<<20),b'')):
            h.update(data);p=cache/f'{i:05d}.part';p.write_bytes(data);manifest.append(dict(name=p.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    digest=h.hexdigest();dest=REMOTE_BASE+'/'+digest
    state=json.loads(remote("import pathlib,json,hashlib\np=pathlib.Path("+repr(dest)+");p.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.part')}))"))
    def one(r):
        if state.get(r['name'])==r['sha256']:return 'REUSED '+r['name']
        error=''
        for attempt in range(4):
            try:
                p=subprocess.run(['scp','-C','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(cache/r['name']),'easydesign-cluster-two:'+dest+'/'+r['name']],capture_output=True,timeout=180)
                if p.returncode==0:return 'SENT '+r['name']
                error=p.stderr.decode()[-500:]
            except subprocess.TimeoutExpired:error='bounded transfer timeout'
            time.sleep(2)
        raise RuntimeError(r['name']+': '+error)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for k,fut in enumerate(concurrent.futures.as_completed([pool.submit(one,r) for r in manifest]),1):print(k,len(manifest),fut.result(),flush=True)
    code=f'''import pathlib,json,hashlib,tarfile,os
p=pathlib.Path({dest!r});manifest={manifest!r};out=p/{f.name!r};h=hashlib.sha256()
with out.open('wb') as stream:
 for r in manifest:
  b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];h.update(b);stream.write(b)
assert h.hexdigest()=={digest!r}
target={a.extract_to!r}
if target:
 target=pathlib.Path(target).resolve();assert str(target).startswith(('/data/PepSeek/data/','/data/PepSeek/runtime/curation/curated_dual_v1_20261013/source_'))
 with tarfile.open(out) as t:
  for m in t.getmembers():
   assert not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts and not m.issym() and not m.islnk()
   if m.isfile():
    b=t.extractfile(m).read();f=target/m.name;f.parent.mkdir(parents=True,exist_ok=True)
    if f.exists():assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(b).digest(),m.name
    else:
     tmp=f.with_name(f.name+'.transfer_tmp');tmp.write_bytes(b);os.replace(tmp,f)
receipt=dict(archive=str(out),sha256=h.hexdigest(),bytes=out.stat().st_size,extracted_to=str(target));(p/'RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    result=remote(code);(cache/'RECEIPT.json').write_text(result);print(result,flush=True)
if __name__=='__main__':main()
