"""Publish finalized clean dataset parts with bounded retries and SHA receipts."""
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--release-id',type=int,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();root=Path(a.package)
    token=os.environ.pop('GITHUB_TOKEN');api='https://api.github.com/repos/S-WCCC/PepSeek';upload=f'https://uploads.github.com/repos/S-WCCC/PepSeek/releases/{a.release_id}/assets'
    def session():
        s=requests.Session();s.trust_env=False;s.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'});s.verify='/etc/ssl/certs/ca-certificates.crt';return s
    s=session();manifest=json.loads((root/'DOWNLOAD_MANIFEST.json').read_text());parts=[p for ar in manifest['archives'] for p in ar['parts']]
    p=root/'DOWNLOAD_MANIFEST.json';parts.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    existing={};page=1
    while True:
        r=s.get(f'{api}/releases/{a.release_id}/assets',params=dict(per_page=100,page=page),timeout=60);r.raise_for_status();arr=r.json();existing.update({x['name']:x for x in arr})
        if len(arr)<100:break
        page+=1
    def one(part):
        name=part['file'];p=root/name;ss=session();last=None
        for attempt in range(4):
            try:
                item=existing.get(name)
                if attempt:
                    for page in range(1,10):
                        rr=ss.get(f'{api}/releases/{a.release_id}/assets',params=dict(per_page=100,page=page),timeout=60);rr.raise_for_status();arr=rr.json();item=next((x for x in arr if x['name']==name),None)
                        if item or len(arr)<100:break
                if item:
                    if item['state']=='uploaded':
                        assert item['size']==part['bytes'] and (not item.get('digest') or item['digest']=='sha256:'+part['sha256']);return dict(part,asset_id=item['id'],url=item['browser_download_url'],api_url=item['url'],status='VERIFIED_EXISTING')
                    # Only an incomplete asset for this exact requested filename.
                    assert item['state']=='starter'
                    rr=ss.delete(item['url'],timeout=60);rr.raise_for_status()
                with p.open('rb') as f:rr=ss.post(upload,params={'name':name},data=f,headers={'Content-Type':'application/octet-stream','Content-Length':str(part['bytes'])},timeout=(30,180))
                rr.raise_for_status();item=rr.json();assert item['size']==part['bytes'];assert item.get('digest') in [None,'sha256:'+part['sha256']]
                return dict(part,asset_id=item['id'],url=item['browser_download_url'],api_url=item['url'],status='UPLOADED_VERIFIED')
            except Exception as err:last=str(err);time.sleep(3+attempt)
        raise RuntimeError(name+' '+str(last))
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(one,p) for p in parts]):
            done.append(f.result());(root/'PUBLISH_RECEIPT.json').write_text(json.dumps(dict(release_id=a.release_id,total=len(parts),completed=len(done),assets=done),indent=2));print('published',len(done),len(parts),done[-1]['file'],flush=True)
    print('ALL_ASSETS_PUBLISHED',flush=True)
if __name__=='__main__':main()
