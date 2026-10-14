"""Concurrent archive parts must never share a temporary download filename."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import threading


def test_concurrent_release_parts_roundtrip(tmp_path, monkeypatch):
    def sha(b):
        return hashlib.sha256(b).hexdigest()
    files={'clean/dataset.json': b'{}\n', 'data/a.bin':os.urandom(70000), 'data/b.bin':os.urandom(70000)}
    manifest=b''.join((json.dumps(dict(path=n,bytes=len(b),sha256=sha(b)))+'\n').encode() for n,b in files.items())
    files['RELEASE_MANIFEST.jsonl']=manifest
    files['RELEASE_SEAL.json']=json.dumps(dict(manifest_sha256=sha(manifest))).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
    blob=buffer.getvalue();cut=len(blob)//2
    assets={f'clean_training.part{i:04d}':b for i,b in enumerate([blob[:cut],blob[cut:]])}
    parts=[dict(file=n,bytes=len(b),sha256=sha(b)) for n,b in assets.items()]
    assets['DOWNLOAD_MANIFEST.json']=json.dumps(dict(archives=[dict(name='clean_training',archive='clean_training.tar.gz',bytes=len(blob),sha256=sha(blob),parts=parts)])).encode()
    barrier=threading.Barrier(2)
    class Part(io.BytesIO):
        first=True
        def read(self,*args):
            if self.first:
                self.first=False;barrier.wait(timeout=10)
            return super().read(*args)
    def open_url(req,**kwargs):
        name=req.full_url.rsplit('/',1)[-1]
        return Part(assets[name]) if '.part' in name else io.BytesIO(assets[name])
    monkeypatch.setattr('urllib.request.urlopen',open_url)
    monkeypatch.setattr(sys,'argv',['download_curated_data.py','--output',str(tmp_path),'--workers','2'])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/download_curated_data.py'),run_name='__main__')
    for name,data in files.items():
        assert (tmp_path/name).read_bytes()==data
