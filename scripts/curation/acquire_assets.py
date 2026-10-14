"""Acquire exact source snapshots on the data disk; accept only matching SHA256.

Coordinate URLs are public archives. A downloaded current revision with a different
hash is retained separately and never silently substituted into the sealed corpus.
"""
import argparse,concurrent.futures,gzip,json,os,shutil,time,urllib.request,urllib.parse,ssl,http.client,threading
from pathlib import Path
from build_catalog import loadjl,save,sha,jsonl

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--workers',type=int,default=20);a=ap.parse_args()
    root=Path(a.root);assets=loadjl(root/'staging/asset_transfer.jsonl');began=time.time()
    ca=os.environ.get('SSL_CERT_FILE')
    context=ssl.create_default_context(cafile=ca);local=threading.local()
    def download(url,path):
        for redirect in range(4):
            u=urllib.parse.urlsplit(url)
            if u.scheme!='https':raise ValueError('HTTPS required')
            if not hasattr(local,'connections'):local.connections={}
            key=(u.hostname,u.port or 443)
            conn=local.connections.get(key)
            if conn is None:conn=http.client.HTTPSConnection(*key,timeout=60,context=context);local.connections[key]=conn
            try:
                conn.request('GET',u.path+('?' + u.query if u.query else ''),headers={'User-Agent':'PepSeek academic data curation/1.0','Connection':'keep-alive'})
                response=conn.getresponse()
                if response.status in (301,302,303,307,308):
                    url=urllib.parse.urljoin(url,response.getheader('Location'));response.read();continue
                if response.status!=200:response.read();raise RuntimeError('HTTP '+str(response.status))
                with path.open('wb') as out:
                    for b in iter(lambda:response.read(1<<20),b''):out.write(b)
                return
            except Exception:
                conn.close();local.connections.pop(key,None);raise
        raise RuntimeError('Redirect limit')
    def one(r):
        p=root/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if p.is_file() and sha(p)==r['sha256']:return dict(path=r['path'],status='VERIFIED_EXISTING',sha256=r['sha256'])
        error='NO_REMOTE_URL'
        for url in r.get('urls',[]):
            for attempt in range(3):
                tmp=p.with_name(p.name+'.partial')
                try:
                    download(url,tmp)
                    h=sha(tmp)
                    if h!=r['sha256']:
                        mismatch=p.with_name(p.name+'.different_'+h[:12]);os.replace(tmp,mismatch);error='SOURCE_REVISION_SHA_MISMATCH';break
                    os.replace(tmp,p)
                    return dict(path=r['path'],status='VERIFIED_DOWNLOADED',sha256=h,source_url=url)
                except Exception as e:error=repr(e);time.sleep(min(2**attempt,4))
        return dict(path=r['path'],status='NEEDS_LOCAL_EXACT_ASSET',reason=error,sha256=r['sha256'])
    counts={};result=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool,(root/'audit/acquisition_stream.jsonl').open('a') as log:
        futures=[pool.submit(one,r) for r in assets]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            r=f.result();result.append(r);counts[r['status']]=counts.get(r['status'],0)+1;log.write(json.dumps(r)+'\n');log.flush()
            if i%100==0:
                save(root/'audit/acquisition_progress.json',dict(done=i,total=len(assets),counts=counts,elapsed_seconds=time.time()-began));print(i,counts,flush=True)
    jsonl(root/'audit/asset_acquisition.jsonl',result);save(root/'audit/ASSET_ACQUISITION_RECEIPT.json',dict(total=len(assets),counts=counts,elapsed_seconds=time.time()-began))
if __name__=='__main__':main()
