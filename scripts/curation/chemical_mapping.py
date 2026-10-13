"""Explicit peptide graph-to-residue mapping using audited annotation grammar.
Adapted from the sealed PPIKB curation grammar; never substitute reconstructed
chemistry for the supplied experimental chemical identity.
"""
import re
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
AA=set('ACDEFGHIKLMNPQRSTVWY')
EMPTY={'','no','none','null','n/a','na','-','unmodified'}
NTERMS={'free':'free','acetyl':'acetyl','acetylated':'acetyl','acetylation':'acetyl'}
CTERMS={'free':'free','amide':'amide','amidated':'amide','amidation':'amide'}
NUM=r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
FLIP={'=':'=','<':'>','>':'<','<=':'>=','>=':'<=','~':'~'}

def canon(m):return Chem.MolToSmiles(m,isomericSmiles=True)
def atom(rw,pos,name):
 z=[a.GetIdx() for a in rw.GetAtoms() if a.GetPDBResidueInfo() and a.GetPDBResidueInfo().GetResidueNumber()==pos and a.GetPDBResidueInfo().GetName().strip()==name]
 if len(z)!=1:raise ValueError('AMBIGUOUS_ATOM')
 return z[0]

@lru_cache(maxsize=16000)
def expected(seq,nt,ct,cyclic,method,way,mod,modified):
 """Small fully specified chemistry grammar. Unhandled text causes rejection."""
 if not seq or not set(seq)<=AA:raise ValueError('NONSTANDARD_PEPTIDE_NOT_FULLY_SPECIFIED')
 if nt not in NTERMS or ct not in CTERMS:raise ValueError('TERMINAL_CHEMISTRY_UNRESOLVED')
 rw=Chem.RWMol(Chem.MolFromSequence(seq));ops=[]
 if CTERMS[ct]=='amide':
  a=rw.GetAtomWithIdx(atom(rw,len(seq),'OXT'));a.SetAtomicNum(7);a.SetFormalCharge(0);a.SetNumExplicitHs(0);a.SetNoImplicit(False);ops.append('C_AMIDE')
 if NTERMS[nt]=='acetyl':
  n=atom(rw,1,'N');c=rw.AddAtom(Chem.Atom(6));o=rw.AddAtom(Chem.Atom(8));ch=rw.AddAtom(Chem.Atom(6))
  rw.AddBond(n,c,Chem.BondType.SINGLE);rw.AddBond(c,o,Chem.BondType.DOUBLE);rw.AddBond(c,ch,Chem.BondType.SINGLE);ops.append('N_ACETYL')
 if mod not in EMPTY:
  parts=[s.strip() for s in re.split(r';',mod) if s.strip()]
  for part in parts:
   m=re.fullmatch(r'([KSTY])(\d+)\s*=\s*(trimethyllysine|dimethyllysine|monomethyllysine|phosphoserine|phosphothreonine|phosphotyrosine)',part,re.I)
   if not m:raise ValueError('MODIFICATION_GRAMMAR_UNRESOLVED')
   aa,pos,name=m[1].upper(),int(m[2]),m[3].lower()
   if not 1<=pos<=len(seq) or seq[pos-1]!=aa:raise ValueError('MODIFICATION_POSITION_CONFLICT')
   if name.endswith('methyllysine'):
    if aa!='K':raise ValueError('MODIFICATION_RESIDUE_CONFLICT')
    n=atom(rw,pos,'NZ');count={'trimethyllysine':3,'dimethyllysine':2,'monomethyllysine':1}[name]
    if count==3:rw.GetAtomWithIdx(n).SetFormalCharge(1)
    for _ in range(count):rw.AddBond(n,rw.AddAtom(Chem.Atom(6)),Chem.BondType.SINGLE)
   else:
    aa_expected={'phosphoserine':'S','phosphothreonine':'T','phosphotyrosine':'Y'}[name]
    if aa!=aa_expected:raise ValueError('MODIFICATION_RESIDUE_CONFLICT')
    o=atom(rw,pos,{'S':'OG','T':'OG1','Y':'OH'}[aa]);p=rw.AddAtom(Chem.Atom(15));rw.AddBond(o,p,Chem.BondType.SINGLE)
    for bond in [Chem.BondType.DOUBLE,Chem.BondType.SINGLE,Chem.BondType.SINGLE]:rw.AddBond(p,rw.AddAtom(Chem.Atom(8)),bond)
   ops.append(f'{aa}{pos}_{name}')
 elif modified not in EMPTY:raise ValueError('DECLARED_MODIFICATION_WITHOUT_IDENTITY')
 if cyclic=='cyclic':
  if 'disulfide' not in method.lower() or (way not in EMPTY and 'disulfide' not in way.lower()):raise ValueError('CYCLIZATION_IDENTITY_UNRESOLVED')
  pairs=re.findall(r'C(?:ys)?\s*(\d+)\s*<?[-=]+>?\s*C(?:ys)?\s*(\d+)',method,re.I)
  residue_positions=re.findall(r'C(?:ys)?\s*(\d+)',method,re.I)
  if not pairs or len(residue_positions)!=2*len(pairs):raise ValueError('CYCLIZATION_POSITIONS_UNRESOLVED')
  used=set()
  for a,b in pairs:
   a,b=int(a),int(b)
   if a==b or a in used or b in used or not 1<=a<=len(seq) or not 1<=b<=len(seq) or seq[a-1]!='C' or seq[b-1]!='C':raise ValueError('DISULFIDE_POSITION_CONFLICT')
   used.update([a,b]);rw.AddBond(atom(rw,a,'SG'),atom(rw,b,'SG'),Chem.BondType.SINGLE);ops.append(f'DISULFIDE_{a}_{b}')
 elif cyclic!='linear' or method not in EMPTY or way not in EMPTY:raise ValueError('LINEAR_CYCLIC_FIELDS_CONFLICT')
 mol=rw.GetMol();Chem.SanitizeMol(mol);return mol,ops


def mapped_identity(entity):
    chem=entity['chemical_identity'];mods=chem['modifications'];seq=entity['sequence']
    mol=Chem.MolFromSmiles(chem['smiles'])
    if mol is None:raise ValueError('Invalid supplied SMILES')
    expected_mol,ops=expected(seq,mods['n_terminal'].lower().strip(),mods['c_terminal'].lower().strip(),mods['linear_cyclic'].lower().strip(),mods['cyclization_method'].strip(),mods['cyclization_way'].lower().strip(),mods['chemical_modification'].strip(),mods['residue_modification'].lower().strip())
    mapping={a.GetIdx():a.GetPDBResidueInfo().GetResidueNumber()-1 for a in expected_mol.GetAtoms() if a.GetPDBResidueInfo()}
    pending=set(range(expected_mol.GetNumAtoms()))-set(mapping)
    while pending:
        updates={}
        for i in pending:
            labels={mapping[n.GetIdx()] for n in expected_mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in mapping}
            if len(labels)==1:updates[i]=labels.pop()
            elif len(labels)>1:raise ValueError('Modification spans ambiguously mapped residues')
        if not updates:raise ValueError('Unmapped chemical component')
        mapping.update(updates);pending-=set(updates)
    left,right=mol,expected_mol
    if canon(left)!=canon(right):
        un=rdMolStandardize.Uncharger();left=un.uncharge(left);right=un.uncharge(right)
        if left.GetNumAtoms()!=mol.GetNumAtoms() or canon(left)!=canon(right):raise ValueError('Supplied graph and annotations disagree')
    match=left.GetSubstructMatch(right,useChirality=True)
    if len(match)!=mol.GetNumAtoms():raise ValueError('Full chemical graph isomorphism failed')
    amap=[-1]*mol.GetNumAtoms()
    for i,j in enumerate(match):amap[j]=mapping[i]
    if set(amap)!=set(range(len(seq))):raise ValueError('Incomplete atom-to-residue map')
    # Keep the supplied canonical SMILES atom order exactly; conformers use this graph.
    return {'smiles':chem['smiles'],'atom_residue':amap,'chemistry_mode':'explicit','chemical_identity_id':entity['id'],'chemical_mapping_rule':'annotation reconstruction + full chiral isomorphism; acid/base normalization only for matching','modification_operations':ops}
