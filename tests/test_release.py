import ast,json,unittest,importlib.util,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 def test_syntax(self):
  for f in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py')):ast.parse(f.read_text(),filename=str(f))
 def test_manifest_counts(self):
  for filename,n in [('samples',37500),('validation_diagnostics_50',50),('fixed_geometry_test_180',180),('auxiliary_test_200',200)]:
   rows=[json.loads(l)for l in(ROOT/'configs/manifests'/f'{filename}.jsonl').open()];self.assertEqual(len(rows),n)
 def test_diagnostic_test_separation(self):
  d=ROOT/'configs/manifests';a=[json.loads(l)for l in(d/'validation_diagnostics_50.jsonl').open()];b=[json.loads(l)for l in(d/'fixed_geometry_test_180.jsonl').open()]
  self.assertFalse({r['anchor_sample_id']for r in a}&{r['source_sample_id']for r in b})
  self.assertEqual(len({r['source_sample_id']for r in b}),36)
  self.assertTrue(all(r['source_class']!=r['target_class']for r in b))
 def test_split_membership(self):
  rows=[json.loads(l)for l in(ROOT/'configs/manifests/samples.jsonl').open()];self.assertEqual(len({r['sample_id']for r in rows}),len(rows))
  self.assertEqual({s:sum(r['split']==s for r in rows)for s in ['train','val','test']},{'train':25000,'val':2500,'test':10000})
 def test_paired_statistics(self):
  import csv
  spec=importlib.util.spec_from_file_location('summary',ROOT/'scripts/evaluation/paired_summary.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with(ROOT/'examples/paired_scores.csv').open()as f:r=m.paired_summary(list(csv.DictReader(f)))
  self.assertAlmostEqual(r['mean'],.0125);self.assertAlmostEqual(r['sample_sd'],statistics.stdev([.015,.01]))
 def test_no_raw_images_or_weights(self):
  self.assertFalse([p for p in ROOT.rglob('*')if p.suffix.lower()in ['.png','.jpg','.tif','.tiff','.npz','.pt','.pth','.safetensors']])
if __name__=='__main__':unittest.main()
