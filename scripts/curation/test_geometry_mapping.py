import unittest
from geometry_helpers import mapping,group_maps

class MappingTests(unittest.TestCase):
    def test_component_union_accepts_streamed_keys(self):
        from assign_splits import Union
        u=Union(str(i) for i in range(4));u.join('0','1');u.join('2','3');u.join('1','2')
        self.assertEqual(len({u.find(str(i)) for i in range(4)}),1);self.assertEqual(u.n[u.find('0')],4)
    def test_exact_fragment(self):
        x=mapping('MABCDEFGHIJK','CDEFG');self.assertEqual(x[0],[3,4,5,6,7])
    def test_ambiguous_positions_are_masked(self):
        s='ITAKPAKTPTSPKEQAIGLSVTFLSFLLPAGWVLYHLDNYKKSX';e='ITAKPAKTPTSPKEQAIGLSVTFLSFLLPAGWVLYHLDNYKKSSAA'
        x=mapping(s,e);self.assertIsNotNone(x);self.assertEqual(x[0][:43],list(range(43)));self.assertTrue(all(v==-1 for v in x[0][43:]))
    def test_exact_observed_nonstandard_chain(self):
        ch={'A':dict(type='polypeptide(L)',sequence='STYAS',monomers=['SER','THR','TYR','ALA','SEP'],atoms=[{'CA':1},{'CA':1},{},{'CA':1},{'CA':1}])}
        x=group_maps(('A',),ch,'STAX');self.assertIsNotNone(x);self.assertEqual(x[0][2],[0,1,-1,2,3])
    def test_standard_mismatch_never_wildcard(self):
        ch={'A':dict(type='polypeptide(L)',sequence='STYAS',monomers=['SER','THR','TYR','ALA','SER'],atoms=[{'CA':1},{'CA':1},{},{'CA':1},{'CA':1}])}
        self.assertIsNone(group_maps(('A',),ch,'STAX'))
    def test_unrelated_chain_rejected(self):self.assertIsNone(mapping('AAAAAAAAAA','WWWWWWWWWW'))
    def test_comma_separated_component_identity(self):
        from structure_labels import resolve
        ch={k:dict(type='polypeptide(L)',sequence=s,monomers=['ALA']*len(s),atoms=[{}]*len(s),auth={k}) for k,s in [('A','ACDEF'),('B','GHIKL')]}
        m=resolve('A,B',['ACDEF','GHIKL'],ch,'label');self.assertEqual([x[0] for x in m],['A','B'])

if __name__=='__main__':unittest.main()
